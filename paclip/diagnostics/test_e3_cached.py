"""Cache-only continuation tests; no pretrained models or GPU are used."""
import json
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np
import torch

from paclip.cli import brain_structure_cached as replay


class CachedTests(unittest.TestCase):
    def test_external_pairs_use_only_header_and_match_canonical_shape(self):
        data = np.zeros((3, 4, 5), np.float32)
        affines = [np.diag([-1., -1., 2., 1.]),
                   np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-2, 0, 0, 0], [0, 0, 0, 1]], float)]
        for affine in affines:
            actual = nib.Nifti1Image(data, affine)
            length = nib.as_closest_canonical(actual).shape[2]
            header_only = SimpleNamespace(shape=data.shape, affine=affine)
            with patch.object(replay.nib, 'load', return_value=header_only):
                dataset = replay.HeaderExternalSlices({'cases': [{'image': 'unused'}]})
            self.assertEqual(dataset.pairs, [(0, i) for i in range(length)])

    def cache_fixture(self, folder):
        records = [{'sample_id': 'normal/a.png', 'label': 0},
                   {'sample_id': 'abnormal/b.png', 'label': 1}]
        class Dataset:
            def __len__(self): return 2
        dataset = Dataset(); dataset.records = records
        cache = {'text': np.zeros((2, 3, 3), np.float32),
                 'context_error': np.zeros((2, 3, 3, 3), np.float32),
                 'context_valid': np.ones((2, 3, 3, 3), bool),
                 'roi': np.ones((2, 3, 3), bool), 'labels': np.array([0, 1]),
                 'ids': np.array([r['sample_id'] for r in records]),
                 'groups': np.array(['FLAIR', 'FLAIR']),
                 'case_ids': np.array(['', '']), 'slice_z': np.array([-1, -1])}
        identity = {'base': 'e1', 'source_sequence': 'FLAIR', 'checkpoint_sha256': 'bound-sha'}
        path = folder/'e1_source_val.npz'
        np.savez_compressed(path, **cache)
        path.with_name(path.stem+'.identity.json').write_text(json.dumps(identity), encoding='utf-8')
        return path, identity, dataset, cache

    def test_bound_cache_rejects_identity_groups_and_nonfinite_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, identity, dataset, cache = self.cache_fixture(Path(tmp))
            loaded = replay.load_bound_cache(path, identity, dataset)
            np.testing.assert_array_equal(loaded['text'], cache['text'])
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                replay.load_bound_cache(path, dict(identity, checkpoint_sha256='other-sha'), dataset)
            cache['groups'][0] = 'T1'
            np.savez_compressed(path, **cache)
            with self.assertRaisesRegex(ValueError, 'groups'):
                replay.load_bound_cache(path, identity, dataset)
            cache['groups'][0] = 'FLAIR'; cache['context_error'][0, 0, 0, 0] = np.nan
            np.savez_compressed(path, **cache)
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                replay.load_bound_cache(path, identity, dataset)

    def fake_session(self, root):
        session = replay.CachedSession.__new__(replay.CachedSession)
        session.root = root/'baseline'; session.temperature = .07
        session.protocols = {'e1': {'checkpoint_sha256': 'sha', 'support_digest': 'support',
                                    'betas': [0., .1], 'froc_points': 4}}
        session.input_hashes = {'cache': 'bound'}
        session.caches = {'e1': {name: {'labels': np.array([0, 1])} for name in replay.DATASETS}}
        session.datasets = {name: SimpleNamespace() for name in replay.DATASETS}
        session.validation = SimpleNamespace()
        session.context = Mock(return_value=SimpleNamespace(calibration={'groups': {'FLAIR': {}}},
                                                            last_group_fallback=0))
        session.base_scores = Mock(return_value=np.zeros((2, 3, 3), np.float32))
        session.index = Mock(return_value=SimpleNamespace(summary={'available': False}))
        return session

    def test_distance_nan_is_only_allowed_in_unsupported_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, identity, dataset, cache = self.cache_fixture(Path(tmp))
            identity['base'] = 'e2'
            path.with_name(path.stem+'.identity.json').write_text(json.dumps(identity), encoding='utf-8')
            supported = torch.ones((3, 3, 3), dtype=torch.bool)
            supported[0, 0, 0] = False
            memory = SimpleNamespace(_support_grid=lambda *args: supported)
            cache['distance'] = np.zeros((2, 3, 3, 3), np.float32)
            cache['distance'][:, 0, 0, 0] = np.nan
            np.savez_compressed(path, **cache)
            replay.load_bound_cache(path, identity, dataset, memory)
            cache['distance'][0, 1, 0, 0] = np.nan
            np.savez_compressed(path, **cache)
            with self.assertRaisesRegex(ValueError, 'supported prototype'):
                replay.load_bound_cache(path, identity, dataset, memory)

    def pipeline_patches(self):
        metrics = {'small_lesion_recall': .5, 'normal_slice_false_positive_components': 0.,
                   'pixel_ap': .7, 'dice_micro': .6}
        row = {'pixel_threshold': .5, 'image_threshold': .5, 'froc_thresholds': [.5],
               'froc_points': 1, 'threshold_source': 'source_val', 'score_space': 'probability',
               'metrics': metrics}
        return patch.multiple(replay.core,
            context_low_scores=Mock(return_value=(np.ones((2, 3, 3), np.float32), np.ones((2, 3, 3), bool))),
            validate_scores=Mock(side_effect=lambda *a, **k: dict(row)),
            representative_indices=Mock(return_value=[]),
            evaluate_scores=Mock(side_effect=lambda *a, **k: {'pixel_ap': .7, 'dice_micro': .6}))

    def test_run_and_completed_skip_never_load_model_or_touch_baselines(self):
        with tempfile.TemporaryDirectory() as tmp, self.pipeline_patches():
            root = Path(tmp); session = self.fake_session(root)
            session.root.mkdir(); sentinel = session.root/'keep.txt'; sentinel.write_text('unchanged')
            with patch.object(replay.core, 'load_trained_model', side_effect=AssertionError('No model allowed')) as loader:
                session.run('C1', root/'output')
                calls = replay.core.evaluate_scores.call_count
                self.assertEqual(calls, 3)
                session.run('C1', root/'output')
                self.assertEqual(replay.core.evaluate_scores.call_count, calls)
                loader.assert_not_called()
            self.assertEqual(sentinel.read_text(), 'unchanged')
            self.assertEqual(json.loads((root/'output/C1/protocol.json').read_text())['status'], 'complete')
            with self.assertRaisesRegex(ValueError, 'separate output'):
                session.run('C1', session.root)

    def test_failed_evaluation_marks_protocol_and_completion_failed(self):
        with tempfile.TemporaryDirectory() as tmp, self.pipeline_patches():
            root = Path(tmp); session = self.fake_session(root)
            replay.core.evaluate_scores.side_effect = RuntimeError('simulated CPU evaluation failure')
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                session.run('C1', root/'output')
            for file in ('protocol.json', 'completion.json'):
                self.assertEqual(json.loads((root/'output/C1'/file).read_text())['status'], 'failed')

    def test_report_merges_original_baselines_and_completed_continuation(self):
        project = Path(replay.__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location('cached_report_test', project/'report_brain_e3.py')
        report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for parent, name, status in [('baseline', 'B1', 'complete'), ('baseline', 'B2', 'complete'),
                                         ('output', 'C1', 'complete'), ('output', 'C2', 'running')]:
                folder = root/parent/name; folder.mkdir(parents=True)
                (folder/'protocol.json').write_text(json.dumps({'status':status,'selected_beta':0.}), encoding='utf-8')
                (folder/'completion.json').write_text(json.dumps({'status':'complete'}), encoding='utf-8')
                (folder/'results.json').write_text(json.dumps({'source_test':{'pixel_ap':.7,'dice_micro':.6}}), encoding='utf-8')
            original = (root/'baseline/B1/protocol.json').read_bytes()
            report.main(['--root', str(root/'output'), '--baseline-root', str(root/'baseline')])
            with (root/'output/comparison.csv').open(encoding='utf-8-sig',newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row['config'] for row in rows}, {'B1','B2','C1'})
            self.assertEqual((root/'baseline/B1/protocol.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
