"""Offline parity: ONNX export + ONNX Runtime match the torch scorer.

Uses a tiny ModernBertConfig (no download), mirroring test_inference.py. Skips
cleanly if onnx/onnxruntime are not installed (they are an optional backend).
"""
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

from models.inference import run_inference
from models.modernbert import ModernBertScorer


def _tiny_tokenizer():
    vocab = {word: i for i, word in enumerate(
        ['[PAD]', '[CLS]', '[SEP]', '[UNK]', 'STATE', 'board', '0', '1',
         'CANDIDATE', 'left', 'right', 'center', 'board_after'])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 1), ('[SEP]', 2)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token='[PAD]', cls_token='[CLS]',
        sep_token='[SEP]', unk_token='[UNK]', padding_side='right')
    return tokenizer, len(vocab)


class OnnxCheck(unittest.TestCase):
    def test_parity_with_torch(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError as error:
            self.skipTest(f"onnxruntime not installed: {error}")

        from models.export_onnx import export
        from models.onnx_scorer import OnnxScorer

        torch.manual_seed(7)
        tokenizer, vocab_size = _tiny_tokenizer()
        config = ModernBertConfig(
            vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2,
            max_position_embeddings=64, local_attention=8,
            pad_token_id=0, cls_token_id=1, bos_token_id=1, eos_token_id=2, sep_token_id=2,
            reference_compile=False, attn_implementation='sdpa')
        # dropout 0 so torch eval and the exported graph are bit-comparable.
        model = ModernBertScorer(ModernBertModel(config), dropout=0.0,
                                 tokenizer_id='tiny')
        state = 'STATE\nboard: 0 0 1\n\nCANDIDATE\n'
        prompts = [state + 'left\nboard_after: 1 0 1',
                   state + 'right\nboard_after: 0 1 1',
                   state + 'center 0 0 1\nboard_after: 0 0 1']
        ids = ['p1', 'p6', 'p9']
        torch_result = run_inference(model, tokenizer, prompts, ids)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'model.onnx'
            export(model, out)  # self-checks logits internally too
            self.assertTrue(out.exists() and Path(str(out) + '.json').exists())
            scorer = OnnxScorer(out)
            self.assertEqual(scorer.tokenizer_id, 'tiny')
            self.assertEqual(next(scorer.parameters()).device.type, 'cpu')

            onnx_result = run_inference(scorer, tokenizer, prompts, ids)
            torch.testing.assert_close(
                torch.tensor(onnx_result['probabilities']),
                torch.tensor(torch_result['probabilities']), rtol=1e-3, atol=1e-4)
            self.assertEqual(onnx_result['chosen_candidate_id'],
                             torch_result['chosen_candidate_id'])

            # Dynamic axes: batch size 1 and a different sequence length must work.
            single = run_inference(scorer, tokenizer, [prompts[0]], ['p1'])
            self.assertEqual(single['probabilities'], [1.0])
            self.assertAlmostEqual(single['logits'][0], onnx_result['logits'][0], places=4)


if __name__ == '__main__':
    unittest.main()
