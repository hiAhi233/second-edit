"""Continue C1/C2/D1/D2 from completed E3 baselines without loading the model.

The B caches already contain text margins and context errors; B2 additionally
contains prototype distances and a small calibrated memory. All scoring and
evaluation use the original E3 functions, with their identities checked first.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from paclip.cli import brain_structure as core
from paclip.cli.brain_memory import ExternalSlices
from paclip.data.datasets import SliceAnomalyDataset, scan_split
from paclip.data.fewshot import file_sha256
from paclip.data.slice_context import build_slice_index
from paclip.engine.consistency import smooth_three_slice
from paclip.engine.trainer import atomic_write_json
from paclip.models.structural_context import ContextConfig, StructuralContext

CONFIGS = {'C1': ('e1', False), 'C2': ('e2', False),
           'D1': ('e1', True), 'D2': ('e2', True)}
DATASETS = ('source_val', 'source_test', 'unseen_wmh', 'unseen_atlas')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class HeaderExternalSlices(ExternalSlices):
    """Same slices as ExternalSlices; building pairs only needs the headers."""
    def __init__(self, protocol):
        self.protocol = protocol
        self.is_wmh = ('WMH' in protocol.get('source_description', '')
                       or 'white matter' in protocol.get('target', '').lower())
        self.pairs = []
        for case, record in enumerate(protocol['cases']):
            image = nib.load(record['image'])
            if len(image.shape) != 3:
                raise ValueError('Cached E3 expects 3D external volumes')
            axes = np.argsort(nib.orientations.io_orientation(image.affine)[:, 0])
            canonical_shape = tuple(image.shape[int(axis)] for axis in axes)
            self.pairs.extend((case, z) for z in range(canonical_shape[2]))
        self.loaded = None
        self.volume = None


class CachedValidationViews:
    """Reuse the 36 validation images across candidates without repeated PNG reads."""
    def __init__(self, dataset):
        self.records = dataset.records
        self.rows = []
        for index in range(len(dataset)):
            item = dataset[index]
            self.rows.append({key: item[key] for key in ('gray01', 'gt_mask_high', 'valid_high') if key in item})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def load_bound_cache(path, expected, dataset, memory=None):
    identity_path = path.with_name(path.stem + '.identity.json')
    identity = read_json(identity_path)
    if identity != expected:
        mismatches = sorted(key for key in set(identity) | set(expected)
                            if identity.get(key) != expected.get(key))
        raise ValueError(f'Cache identity mismatch: {path}; fields={mismatches}')
    with np.load(path, allow_pickle=False) as data:
        cache = {key: data[key] for key in data.files}
    needed = {'text', 'context_error', 'context_valid', 'roi', 'labels', 'ids', 'groups', 'case_ids', 'slice_z'}
    if expected['base'] == 'e2':
        needed.add('distance')
    if not needed.issubset(cache):
        raise ValueError(f'Incomplete context cache: {path}')
    n = len(dataset)
    if any(len(value) != n for value in cache.values()) or len(set(cache['ids'].tolist())) != n:
        raise ValueError(f'Cache sample counts/IDs are invalid: {path}')
    if cache['text'].ndim != 3:
        raise ValueError(f'Invalid text grid: {path}')
    hw = cache['text'].shape[-2:]
    if (cache['roi'].shape != (n, *hw)
            or cache['context_error'].shape != (n, 3, *hw)
            or cache['context_valid'].shape != cache['context_error'].shape):
        raise ValueError(f'Invalid context grid: {path}')
    if cache['roi'].dtype != np.bool_ or cache['context_valid'].dtype != np.bool_:
        raise ValueError(f'ROI/context validity masks must be boolean: {path}')
    for key in ('text', 'context_error'):
        if key in cache and not np.isfinite(cache[key]).all():
            raise ValueError(f'Non-finite {key}: {path}')
    if 'distance' in cache and cache['distance'].shape != cache['context_error'].shape:
        raise ValueError(f'Invalid prototype distance grid: {path}')
    if 'distance' in cache:
        if memory is None:
            raise ValueError('Distance validation requires the bound prototype memory')
        supported = memory._support_grid(hw, 'cpu').numpy()
        distance = cache['distance']
        # Empty prototype regions deliberately carry NaN and fall back to text.
        if np.isinf(distance).any() or not np.isfinite(distance[:, supported]).all():
            raise ValueError(f'Non-finite distance in a supported prototype region: {path}')
        if not np.isnan(distance[:, ~supported]).all():
            raise ValueError(f'Unexpected distance in an unsupported prototype region: {path}')
    if not np.isin(cache['labels'], [0, 1]).all():
        raise ValueError(f'Invalid labels: {path}')
    ids, groups, cases, zs = [], [], [], []
    for index in range(n):
        case, z = core.slice_info(dataset, index)
        cases.append(case); zs.append(z)
        ids.append(f'{case}/z{z:04d}' if isinstance(dataset, ExternalSlices)
                   else dataset.records[index]['sample_id'])
        groups.append(core.sample_group(dataset, index, expected['source_sequence']))
    for key, values in [('ids', ids), ('groups', groups), ('case_ids', cases), ('slice_z', zs)]:
        if not np.array_equal(cache[key], np.asarray(values)):
            raise ValueError(f'Cache {key} do not match the current dataset: {path}')
    if not isinstance(dataset, ExternalSlices):
        if not np.array_equal(cache['labels'], [record['label'] for record in dataset.records]):
            raise ValueError(f'Cache labels do not match the source split: {path}')
    return cache


class CachedSession:
    def __init__(self, baseline_root, temperature=.07):
        self.root = Path(baseline_root).resolve()
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be finite and positive')
        self.temperature = float(temperature)
        self.protocols = {}
        for name, base in [('B1', 'e1'), ('B2', 'e2')]:
            protocol = read_json(self.root/name/'protocol.json')
            completion = read_json(self.root/name/'completion.json')
            if protocol.get('status') != 'complete' or completion.get('status') != 'complete':
                raise ValueError(f'{name} must be completed before a cached continuation')
            if (protocol.get('config') != name or protocol.get('base') != base
                    or protocol.get('context') != 'off' or protocol.get('consistency') != 'off'):
                raise ValueError(f'{name} is not the expected disabled baseline')
            if set(read_json(self.root/name/'results.json')) != set(DATASETS[1:]):
                raise ValueError(f'{name} does not contain all three completed test datasets')
            self.protocols[base] = protocol
        p = self.protocols['e1']
        common = ('checkpoint_sha256', 'support_digest', 'data_root', 'mask_root', 'source_sequence',
                  'alpha', 'betas', 'context_config', 'froc_points', 'unseen_protocols', 'code_fingerprint', 'cache_root')
        if any(p.get(key) != self.protocols['e2'].get(key) for key in common):
            raise ValueError('B1 and B2 were produced with different experiment settings')
        if p['code_fingerprint'] != core.code_fingerprint():
            raise ValueError('Feature/evaluation code changed; existing B caches cannot be silently reused')
        support = read_json(p['support_manifest'])
        if support.get('support_digest') != p['support_digest']:
            raise ValueError('Support manifest does not match the completed baselines')
        self.memory = core.load_base_memory(self.root/'B2/normal_memory.pt',
                                           p['checkpoint_sha256'], p['support_digest'])
        self.context_config = ContextConfig(**p['context_config'])
        self.datasets = {}
        for name, split in [('source_val', 'val'), ('source_test', 'test')]:
            data = Path(p['data_root'])/split; masks = Path(p['mask_root'])/split
            self.datasets[name] = SliceAnomalyDataset(str(data), records=scan_split(str(data), str(masks)))
        for name in DATASETS[2:]:
            provenance = read_json(self.root/'B1'/f'{name}_provenance.json')
            if provenance != read_json(self.root/'B2'/f'{name}_provenance.json'):
                raise ValueError(f'B1/B2 external provenance mismatch: {name}')
            self.datasets[name] = HeaderExternalSlices(provenance)
        self.caches, self.input_hashes = {}, {}
        prompt_digest = None
        for base in ('e1', 'e2'):
            self.caches[base] = {}
            for name in DATASETS:
                dataset = self.datasets[name]
                path = Path(p['cache_root'])/f'{base}_{name}.npz'
                saved = read_json(path.with_name(path.stem + '.identity.json'))
                if prompt_digest is None:
                    prompt_digest = saved.get('prompt_digest')
                if not prompt_digest or saved.get('prompt_digest') != prompt_digest:
                    raise ValueError('Cache text prompts are missing or inconsistent')
                expected = {
                    'version': 'E3_features_v1', 'dataset': name, 'base': base,
                    'checkpoint_sha256': p['checkpoint_sha256'], 'support_digest': p['support_digest'],
                    'prompt_digest': prompt_digest, 'context': asdict(self.context_config),
                    'source_sequence': p['source_sequence'],
                    'memory_fingerprint': core.memory_fingerprint(self.memory) if base == 'e2' else None,
                    'roi': {'threshold': 20./255., 'min_patches': 5, 'empty': 'invalid'},
                    'dataset_fingerprint': core.dataset_fingerprint(dataset),
                    'code_fingerprint': core.code_fingerprint(), 'n_slices': len(dataset)}
                self.caches[base][name] = load_bound_cache(
                    path, expected, dataset, self.memory if base == 'e2' else None)
                self.input_hashes[str(path)] = file_sha256(path)
                self.input_hashes[str(path.with_name(path.stem+'.identity.json'))] = file_sha256(path.with_name(path.stem+'.identity.json'))
                print(f'[cache-only] verified {base}/{name}: {len(dataset)} slices', flush=True)
        for name in ('B1', 'B2'):
            for file in ('protocol.json', 'completion.json', 'validation_search.json', 'results.json',
                         'unseen_wmh_provenance.json', 'unseen_atlas_provenance.json'):
                path = self.root/name/file
                self.input_hashes[str(path)] = file_sha256(path)
        self.input_hashes[str(self.root/'B2/normal_memory.pt')] = file_sha256(self.root/'B2/normal_memory.pt')
        self.input_hashes[str(Path(p['support_manifest']))] = file_sha256(p['support_manifest'])
        self.validation = CachedValidationViews(self.datasets['source_val'])
        self.contexts = {}
        self.replay = self.check_baseline_replay()
        self.indexes = {}

    def base_scores(self, base, name):
        return core.base_low_scores(self.caches[base][name], base,
                                    self.memory if base == 'e2' else None,
                                    self.protocols[base]['alpha'], self.temperature)

    def check_baseline_replay(self):
        replay = {}
        for base, tag in [('e1', 'B1'), ('e2', 'B2')]:
            actual = core.validate_scores(self.base_scores(base, 'source_val'),
                                          self.caches[base]['source_val'], self.validation,
                                          self.protocols[base]['froc_points'])
            expected = read_json(self.root/tag/'validation_search.json')['candidates'][0]
            for key in ('pixel_threshold', 'image_threshold', 'froc_thresholds'):
                if not np.allclose(actual[key], expected[key], rtol=1e-6, atol=1e-12):
                    raise ValueError(f'Baseline {tag} {key} replay failed; check temperature and caches')
            keys = ('pixel_ap', 'pixel_auroc', 'dice_micro', 'lesion_recall', 'small_lesion_recall',
                    'normal_false_positive_area_ratio', 'normal_slice_false_positive_components')
            differences = {key: actual['metrics'][key]-expected['metrics'][key] for key in keys}
            if any(not np.isfinite(value) or abs(value) > 1e-9 for value in differences.values()):
                raise ValueError(f'Baseline {tag} validation replay mismatch: {differences}')
            replay[tag] = {'metric_differences': differences, 'temperature_verified': self.temperature}
            print(f'[cache-only] {tag} validation replay passed', flush=True)
        return replay

    def context(self, base):
        if base not in self.contexts:
            cache = self.caches[base]['source_val']; normal = cache['labels'] == 0
            context = StructuralContext(copy.deepcopy(self.context_config), self.memory.layer_ids)
            context.calibrate(torch.from_numpy(cache['context_error'][normal]),
                              torch.from_numpy(cache['context_valid'][normal]),
                              cache['groups'][normal], cache['ids'][normal].tolist())
            self.contexts[base] = context
        return self.contexts[base]

    def index(self, name):
        if name not in self.indexes:
            self.indexes[name] = build_slice_index(self.datasets[name], name)
        return self.indexes[name]

    def run(self, config, output_root):
        base, consistency = CONFIGS[config]
        output = Path(output_root).resolve()/config
        if output.parent == self.root or self.root in output.parents:
            raise ValueError('Use a separate output root; original baseline/partial directories are read-only')
        identity = {'version': 'E3_cached_v1', 'config': config, 'base': base, 'context': 'on',
                    'consistency': 'on' if consistency else 'off', 'baseline_root': str(self.root),
                    'checkpoint_sha256': self.protocols[base]['checkpoint_sha256'],
                    'support_digest': self.protocols[base]['support_digest'],
                    'temperature': self.temperature, 'source_protocol': self.protocols[base],
                    'input_hashes': self.input_hashes, 'cached_runner_sha256': file_sha256(__file__)}
        protocol_path = output/'protocol.json'
        if protocol_path.exists():
            saved = read_json(protocol_path)
            if saved.get('request_identity') != identity:
                raise ValueError(f'Output identity mismatch: {output}; choose a new output root')
            if (saved.get('status') == 'complete' and (output/'completion.json').exists()
                    and read_json(output/'completion.json').get('status') == 'complete'):
                results = read_json(output/'results.json')
                if set(results) != set(DATASETS[1:]):
                    raise ValueError(f'Completed output is incomplete: {output}')
                print(f'[cache-only] skip completed {config}', flush=True)
                return results
        output.mkdir(parents=True, exist_ok=True)
        protocol = {'request_identity': identity, 'version': 'E3_cached_v1', 'config': config,
                    'base': base, 'context': 'on', 'consistency': 'on' if consistency else 'off',
                    'status': 'running', 'checkpoint_sha256': identity['checkpoint_sha256'],
                    'support_digest': identity['support_digest'], 'no_model_loaded': True}
        atomic_write_json(str(protocol_path), protocol)
        atomic_write_json(str(output/'completion.json'), {'status': 'running', 'config': config})
        started = time.monotonic()
        try:
            context = self.context(base)
            atomic_write_json(str(output/'context_calibration.json'), context.calibration)
            cache = self.caches[base]['source_val']
            base_val = self.base_scores(base, 'source_val')
            context_val, valid = core.context_low_scores(cache, context)
            candidates = []
            for beta in self.protocols[base]['betas']:
                fused = core.fuse_context_scores(base_val, context_val, valid, beta)
                summary = None
                if consistency:
                    fused, summary = smooth_three_slice(fused, self.index('source_val'))
                row = core.validate_scores(fused, cache, self.validation, self.protocols[base]['froc_points'])
                row.update(beta=beta, consistency=summary); candidates.append(row)
            best = core.select_context_candidate(candidates)
            atomic_write_json(str(output/'validation_search.json'), {
                'selection': 'keep small-lesion recall and normal FP <= baseline, then max AP/Dice',
                'baseline_beta': 0., 'selected_beta': best['beta'], 'candidates': candidates})
            print(f'[cache-only] {config} selected beta={best["beta"]}', flush=True)
            results, indexes = {}, {}
            for name in DATASETS[1:]:
                print(f'[cache-only] {config}/{name}: CPU evaluation, no model forward', flush=True)
                dataset = self.datasets[name]; cache = self.caches[base][name]
                base_low = self.base_scores(base, name)
                memory_fallback = self.memory.last_group_fallback if base == 'e2' else 0
                context_low, valid = core.context_low_scores(cache, context)
                final_low = core.fuse_context_scores(base_low, context_low, valid, best['beta'])
                summary = None; index = self.index(name); indexes[name] = index.summary
                if consistency:
                    final_low, summary = smooth_three_slice(final_low, index)
                operating = {key: best[key] for key in ('pixel_threshold', 'image_threshold', 'froc_thresholds',
                                                        'froc_points', 'threshold_source', 'score_space')}
                operating.update(beta=best['beta'], config=config, base=base, context='on',
                                 consistency='on' if consistency else 'off')
                destination = output/name/config
                metrics = core.evaluate_scores(final_low, cache, dataset, operating, destination,
                                               core.representative_indices(dataset, None))
                metrics.update(slice_context=index.summary, consistency=summary,
                               context_group_fallback_slices=int(context.last_group_fallback),
                               base_group_fallback_slices=int(memory_fallback))
                if isinstance(dataset, ExternalSlices):
                    metrics['volume_detection'] = core.volume_metrics(destination, name, dataset)
                    atomic_write_json(str(output/f'{name}_provenance.json'), dataset.protocol)
                atomic_write_json(str(destination/'metrics.json'), metrics)
                results[name] = metrics
                atomic_write_json(str(output/'results.json'), results)
            protocol.update(status='complete', selected_beta=best['beta'], slice_indexes=indexes)
            atomic_write_json(str(protocol_path), protocol)
            atomic_write_json(str(output/'completion.json'), {
                'status': 'complete', 'config': config, 'selected_beta': best['beta'],
                'elapsed_seconds': time.monotonic()-started, 'no_model_loaded': True,
                'slice_indexes': indexes})
            return results
        except BaseException as exc:
            protocol.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            atomic_write_json(str(protocol_path), protocol)
            atomic_write_json(str(output/'completion.json'), {
                'status': 'failed', 'config': config, 'error': protocol['error'],
                'elapsed_seconds': time.monotonic()-started, 'no_model_loaded': True})
            raise


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', default=str(root/'runs/brain_mri_E3_20261025'))
    parser.add_argument('--out-root', default=str(root/'runs/brain_mri_E3_cached_20261025'))
    parser.add_argument('--configs', nargs='+', choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument('--temperature', type=float, default=.07,
                        help='Must reproduce the saved B1/B2 validation thresholds and metrics')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check-only', action='store_true', help='Verify all caches and baseline validation; do not run C/D')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    session = CachedSession(args.baseline_root, args.temperature)
    if args.check_only:
        print('[cache-only] all input checks and baseline validation passed; no C/D evaluation started', flush=True)
        return session.replay
    results = {}
    for config in args.configs:
        results[config] = session.run(config, args.out_root)
    print(f'[cache-only] complete: {Path(args.out_root).resolve()}', flush=True)
    return results


if __name__ == '__main__':
    main()
