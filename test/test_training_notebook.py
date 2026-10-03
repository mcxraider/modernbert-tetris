"""CPU regression checks for notebook data isolation and training failure paths."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def notebook_namespace():
    notebook = json.loads((ROOT / 'model_training/model_training.ipynb').read_text())
    ns = dict(Path=Path, math=math, random=random, json=json, torch=torch)
    exec(''.join(notebook['cells'][3]['source']), ns)
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] != 'code' or source.startswith('%'):
            continue
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(tree, 'training_notebook', 'exec'), ns)
    return ns


class Tokenizer:
    def __call__(self, prompts, **kwargs):
        return dict(input_ids=[[1, 2] for _ in prompts], attention_mask=[[1, 1] for _ in prompts])


def state(sid, episode, targets=(.7, .3), text=None, stage=1):
    return dict(state_id=sid, episode_id=episode, targets=list(targets), state_text=text or sid,
                stage=stage, original_mass=1.)


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.ns = notebook_namespace()
        self.ns['tokenizer'] = Tokenizer()

    def test_episode_and_text_isolation_across_rounds(self):
        first = [state(f'h{i}', f'e{i}') for i in range(120)]
        second = [state(f'j{i}', f'e{i}') for i in range(120)]
        first += [state(f'single{i}', f'e{i}', (1.,)) for i in range(120)]
        # Same text in different episodes and rounds must never cross splits.
        for row in first[:30] + second[30:60]:
            row['state_text'] = 'repeated board'
        datasets = dict(heuristic=first, jev=second)
        rounds, manifests = self.ns['split_rounds'](datasets)
        self.assertEqual((rounds, manifests), self.ns['split_rounds'](datasets))
        episodes, texts = {}, {}
        for stages in rounds.values():
            for splits in stages.values():
                for split, rows in splits.items():
                    for row in rows:
                        self.assertEqual(episodes.setdefault(row['episode_id'], split), split)
                        self.assertEqual(texts.setdefault(row['state_text'], split), split)
                        if split == 'train':
                            self.assertGreater(len(row['targets']), 1)
        self.assertTrue(manifests['heuristic']['excluded_training_singletons'])
        self.assertTrue(manifests['heuristic']['removed_cross_split_text'])

    def test_no_state_split_fallback_for_legacy_bucket(self):
        rows = [state(f's{i}', 'one-bucket-run') for i in range(100)]
        with self.assertRaisesRegex(ValueError, 'independent episodes'):
            self.ns['split_rounds'](dict(heuristic=rows))

    def test_probability_provenance_and_rounding(self):
        def candidate(cid, q, **extra):
            return dict(candidate_id=cid, prompt=f'STATE\nboard\n\nCANDIDATE\n{cid}', probability=q,
                        offered_to_jev=True, probability_observed=True, **extra)
        rows = [candidate('a', .99), candidate('b', 0)]
        group = dict(state_id='s', episode_id='e', stage=1, candidates=rows)
        prepared = self.ns['prepare_states']([group], jev=True)[0]
        self.assertEqual(prepared['targets'], [1., 0.]) # evaluated zero is retained
        self.assertEqual(prepared['original_mass'], .99)
        rows[1]['probability_observed'] = False
        with self.assertRaisesRegex(ValueError, 'incomplete teacher'):
            self.ns['prepare_states']([group], jev=True)
        del rows[1]['probability_observed']
        with self.assertRaisesRegex(ValueError, 'unknown label'):
            self.ns['prepare_states']([group], jev=True)
        rows[1]['eligibility_source'] = 'legacy_solo_offered_only'
        self.ns['prepare_states']([group], jev=True)
        rows[0]['probability'] = .95
        with self.assertRaisesRegex(ValueError, 'probability mass'):
            self.ns['prepare_states']([group], jev=True)
        rows[0]['probability'] = .99
        del rows[0]['offered_to_jev']
        with self.assertRaisesRegex(ValueError, 'offered_to_jev'):
            self.ns['prepare_states']([group], jev=True)

    def test_amp_overflow_recovers_and_does_not_advance_lr(self):
        p = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.SGD([p], lr=.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
        scaler = torch.amp.GradScaler('cpu')
        initial_step = scheduler.last_epoch
        scaler.scale(p * torch.tensor(float('inf'))).backward()
        self.assertFalse(self.ns['amp_optimizer_step'](scaler, optimizer, scheduler))
        self.assertEqual(p.item(), 1.)
        self.assertEqual(scheduler.last_epoch, initial_step)
        self.assertEqual(scaler.get_scale(), 32768.)
        optimizer.zero_grad()
        scaler.scale(p * 2).backward()
        self.assertTrue(self.ns['amp_optimizer_step'](scaler, optimizer, scheduler))
        self.assertAlmostEqual(p.item(), .8, places=6)
        self.assertEqual(scheduler.last_epoch, initial_step+1)

    def test_failed_attempt_cannot_overwrite_artifacts(self):
        def fail_model():
            raise RuntimeError('simulated model-load failure')
        self.ns['fresh_model'] = fail_model
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'attempt'
            manifest = dict(original='split')
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                self.ns['train_round']({}, out, 'test', manifest)
            self.assertTrue((out / 'STARTED').exists())
            before = (out / 'split.json').read_bytes()
            (out / 'baseline.pt').write_bytes(b'preserve baseline')
            with self.assertRaises(FileExistsError):
                self.ns['train_round']({}, out, 'test', dict(replacement=True))
            self.assertEqual((out / 'split.json').read_bytes(), before)
            self.assertEqual((out / 'baseline.pt').read_bytes(), b'preserve baseline')
            self.assertFalse((out / 'COMPLETE').exists())

    def test_decision_metrics_do_not_count_forced_moves(self):
        self.ns['model'] = torch.nn.Linear(1, 1)
        self.ns['make_loader'] = lambda data, **kwargs: data
        self.ns['state_metrics'] = lambda group: (torch.tensor(group['loss']), group['accuracy'])
        metrics = self.ns['evaluate']([dict(targets=[1], loss=0., accuracy=1.),
                                       dict(targets=[.7, .3], loss=2., accuracy=0.)])
        self.assertEqual(metrics['accuracy'], 0.)
        self.assertEqual(metrics['loss'], 2.)
        self.assertEqual(metrics['all_state_accuracy'], .5)
        self.assertEqual(metrics['singletons'], 1)

    def test_plan_visits_each_state_once_and_handles_partial_batches(self):
        stages = {1: dict(train=list(range(43))), 2: dict(train=list(range(71)))}
        plan = self.ns['make_plan'](stages)
        for stage, splits in stages.items():
            indices = [i for s, ids in plan if s == stage for i in ids]
            self.assertEqual(sorted(indices), list(range(len(splits['train']))))
        self.assertTrue(all(0 < len(ids) <= 32 for _, ids in plan))

    def test_export_preserves_known_solo_provenance_only(self):
        spec = importlib.util.spec_from_file_location('collapse', ROOT / 'data/collapse_battles.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for run in ['solo-jev-run', 'battle-jev-run']:
                row = dict(prompt='p', probability=0., run=run)
                (directory / f'{run}.json').write_text(json.dumps({run: {'p0': row}}))
            rows = {s['state_id']: s for s in module.export_states(directory)}
            solo = rows['solo-jev-run']
            self.assertEqual(solo['episode_id'], 'solo-jev-run')
            self.assertTrue(solo['candidates'][0]['offered_to_jev'])
            self.assertNotIn('offered_to_jev', rows['battle-jev-run']['candidates'][0])


if __name__ == '__main__':
    unittest.main()
