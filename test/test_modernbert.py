"""Run from repository root: python -m unittest discover -s test -p 'test_*.py'."""
import math
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import ModernBertConfig, ModernBertModel
from models.modernbert import ModernBertScorer, candidate_probabilities, distillation_loss


class ModelCheck(unittest.TestCase):
    def test_model_and_objective(self):
        torch.manual_seed(7)
        config = ModernBertConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2,
            max_position_embeddings=32, local_attention=8,
            pad_token_id=0, cls_token_id=1, bos_token_id=1, eos_token_id=2, sep_token_id=2,
            reference_compile=False, attn_implementation="sdpa",
        )
        model = ModernBertScorer(ModernBertModel(config), tokenizer_id="test-tokenizer")
        ids = torch.tensor([[1, 3, 2, 0], [1, 4, 5, 2], [1, 6, 2, 0], [1, 7, 2, 0]])
        mask = ids.ne(0)
        logits = model(ids, mask)
        self.assertEqual(logits.shape, (4,))
        self.assertTrue(torch.isfinite(logits).all())
        loss = distillation_loss(logits, torch.tensor([.94, .06, .2, .8]), [2, 2])
        loss.backward()
        for params in (model.encoder.parameters(), model.head.parameters()):
            grads = [p.grad for p in params if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        model.eval()
        with torch.no_grad():
            expected = model(ids, mask)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                model.save_checkpoint(path)
                restored = ModernBertScorer.load_checkpoint(path).eval()
                torch.testing.assert_close(restored(ids, mask), expected)
                self.assertEqual(restored.tokenizer_id, "test-tokenizer")
                # Exercise pretrained construction without downloading a large model.
                encoder_path = Path(directory) / "encoder"
                model.encoder.save_pretrained(encoder_path)
                initialized = ModernBertScorer.from_pretrained(encoder_path)
                for name, value in model.encoder.state_dict().items():
                    torch.testing.assert_close(initialized.encoder.state_dict()[name], value)

        scores = torch.tensor([math.log(.94 / .06), 0., -2., 0., 10.], requires_grad=True)
        eligible = torch.tensor([True, True, True, True, False])
        targets = torch.tensor([.94, .06, .25, .75, 0.])
        probs = candidate_probabilities(scores, [2, 3], eligible)
        torch.testing.assert_close(probs[:2], targets[:2])
        torch.testing.assert_close(probs[2:4], torch.tensor([-2., 0.]).softmax(0))
        self.assertEqual(probs[-1].item(), 0.)
        shifted = scores.detach() + torch.tensor([100., 100., -10., -10., -10.])
        torch.testing.assert_close(candidate_probabilities(shifted, [2, 3], eligible), probs)
        expected_loss = -(targets[:2] * probs[:2].log()).sum() / 2
        expected_loss -= (targets[2:4] * probs[2:4].log()).sum() / 2
        actual = distillation_loss(scores, targets, [2, 3], eligible)
        torch.testing.assert_close(actual, expected_loss)
        actual.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(scores.grad[-1].item(), 0.)
        self.assertTrue(torch.isfinite(distillation_loss(torch.tensor([0., -1.]), torch.tensor([1., 0.]), [2])))
        torch.testing.assert_close(candidate_probabilities(torch.tensor([0.]), [1]), torch.ones(1))

        bad_calls = [
            lambda: candidate_probabilities(scores, [2, 2]),
            lambda: candidate_probabilities(scores, [0, 5]),
            lambda: candidate_probabilities(scores, [5], torch.zeros(5, dtype=torch.bool)),
            lambda: candidate_probabilities(torch.tensor([float('nan')]), [1]),
            lambda: distillation_loss(scores, targets * 0, [2, 3], eligible),
            lambda: distillation_loss(scores, -targets, [2, 3], eligible),
            lambda: distillation_loss(scores, torch.ones(5), [2, 3], eligible),
            lambda: model(ids, torch.ones(4, 3)),
            lambda: model(ids, torch.tensor([[0, 1, 1, 0]] * 4)),
            lambda: model(ids, torch.tensor([[1, 0, 1, 0]] * 4)),
            lambda: model(torch.zeros_like(ids), mask),
        ]
        for call in bad_calls:
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
