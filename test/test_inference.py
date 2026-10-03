"""Offline text -> tokenizer -> real ModernBERT -> softmax/choice -> CLI checks."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

from models.inference import run_inference
from models.modernbert import ModernBertScorer


class InferenceCheck(unittest.TestCase):
    def test_end_to_end(self):
        torch.manual_seed(7)
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
        config = ModernBertConfig(
            vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2,
            max_position_embeddings=64, local_attention=8,
            pad_token_id=0, cls_token_id=1, bos_token_id=1, eos_token_id=2, sep_token_id=2,
            reference_compile=False, attn_implementation='sdpa')
        model = ModernBertScorer(ModernBertModel(config), dropout=.9)
        state = 'STATE\nboard: 0 0 1\n\nCANDIDATE\n'
        prompts = [state + 'left\nboard_after: 1 0 1',
                   state + 'right\nboard_after: 0 1 1',
                   state + 'center 0 0 1\nboard_after: 0 0 1']
        ids = ['p1', 'p6', 'p9']
        calls = []
        hook = model.register_forward_pre_hook(
            lambda module, args: calls.append((args[0].shape[0], torch.is_grad_enabled(), module.training)))
        result = run_inference(model, tokenizer, prompts, ids)
        hook.remove()
        self.assertEqual(calls, [(3, False, False)])
        self.assertEqual(result['candidate_ids'], ids)
        self.assertAlmostEqual(sum(result['probabilities']), 1., places=6)
        self.assertTrue(all(p > 0 for p in result['probabilities']))
        expected = torch.tensor(result['logits']).softmax(0)
        torch.testing.assert_close(torch.tensor(result['probabilities']), expected)
        chosen = expected.argmax().item()
        self.assertEqual(result['chosen_index'], chosen)
        self.assertEqual(result['chosen_candidate_id'], ids[chosen])
        self.assertEqual(result['chosen_prompt'], prompts[chosen])
        self.assertEqual(run_inference(model, tokenizer, prompts, ids), result)
        reversed_result = run_inference(model, tokenizer, prompts[::-1], ids[::-1])
        torch.testing.assert_close(torch.tensor(reversed_result['probabilities'][::-1]), expected)
        # torch.compile must hold fullgraph, keep the same choice, and match numerics (fused
        # kernels differ only at ~1e-6, so compare probabilities with a tolerance, choice exactly).
        compiled = ModernBertScorer(model.encoder, dropout=.9).compile_scorer()
        compiled.load_state_dict(model.state_dict())
        compiled_result = run_inference(compiled, tokenizer, prompts, ids)
        torch.testing.assert_close(torch.tensor(compiled_result['probabilities']), expected)
        self.assertEqual(compiled_result['chosen_candidate_id'], result['chosen_candidate_id'])
        for i, prompt in enumerate(prompts):
            single = run_inference(model, tokenizer, [prompt])
            self.assertEqual(single['probabilities'], [1.])
            self.assertAlmostEqual(single['logits'][0], result['logits'][i], places=5)
        masked = run_inference(model, tokenizer, prompts, ids, [False, True, False])
        self.assertEqual(masked['probabilities'], [0., 1., 0.])
        self.assertEqual(masked['chosen_candidate_id'], 'p6')

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tokenizer.save_pretrained(directory / 'tokenizer')
            model.tokenizer_id = str(directory / 'tokenizer')
            model.save_checkpoint(directory / 'model.pt')
            rows = [dict(state_id='state-1', candidate_id=i, prompt=p, probability=0.)
                    for i, p in zip(ids, prompts)]
            # Teacher target fields must not change inference behavior.
            rows.append(dict(state_id='state-2', candidate_id='other', prompt=prompts[0]))
            (directory / 'input.json').write_text(json.dumps(rows))
            command = [sys.executable, '-m', 'models.inference', str(directory / 'input.json'),
                       '--checkpoint', str(directory / 'model.pt')]
            proc = subprocess.run(command + ['--state-id', 'state-1'],
                                  capture_output=True, text=True, check=True)
            cli = json.loads(proc.stdout)
            for key, value in result.items():
                self.assertEqual(cli[key], value)
            self.assertEqual(cli['source'], 'checkpoint')
            self.assertEqual(cli['state_id'], 'state-1')
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 2)
            self.assertIn('multiple states', rejected.stderr)
            nested = {'state-1': {i: {'prompt': p, 'probability': 0.} for i, p in zip(ids, prompts)},
                      'state-2': {'other': {'prompt': prompts[0]}}}
            (directory / 'input.json').write_text(json.dumps(nested))
            grouped_cli = json.loads(subprocess.run(command + ['--state-id', 'state-1'],
                                                   capture_output=True, text=True, check=True).stdout)
            self.assertEqual(grouped_cli, cli)
            missing = subprocess.run(command + ['--state-id', 'missing'], capture_output=True, text=True)
            self.assertEqual(missing.returncode, 2)
            nested.pop('state-2')
            (directory / 'input.json').write_text(json.dumps(nested))
            auto = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
            self.assertEqual(auto, cli)
            (directory / 'input.json').write_text(json.dumps(prompts))
            plain = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
            self.assertEqual(plain['probabilities'], result['probabilities'])
            self.assertEqual(plain['chosen_candidate_id'], chosen)

        bad_calls = [
            lambda: run_inference(model, tokenizer, []),
            lambda: run_inference(model, tokenizer, 'not a list'),
            lambda: run_inference(model, tokenizer, ['STATE\nboard']),
            lambda: run_inference(model, tokenizer, [prompts[0], prompts[1].replace('board: 0', 'board: 1')]),
            lambda: run_inference(model, tokenizer, prompts, ['duplicate'] * 3),
            lambda: run_inference(model, tokenizer, prompts, ids, [False] * 3),
            lambda: run_inference(model, tokenizer, prompts, ids, [1, 1, 1]),
            lambda: run_inference(model, tokenizer, [state + 'left ' * 100]),
        ]
        for call in bad_calls:
            with self.assertRaises(ValueError):
                call()
        # Zero logits are legitimate; ties select the first eligible candidate.
        with torch.no_grad():
            model.head.weight.zero_()
        uniform = run_inference(model, tokenizer, prompts, ids)
        torch.testing.assert_close(torch.tensor(uniform['probabilities']), torch.full((3,), 1 / 3))
        self.assertEqual(uniform['chosen_candidate_id'], ids[0])


if __name__ == '__main__':
    unittest.main()
