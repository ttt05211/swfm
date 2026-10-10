"""Standalone all-window origin check. Does not initialize a GPU or model."""
import argparse
import json
from pathlib import Path
import time

from real_motion.planner_origin_audit import audit, sha256, summary
from real_motion.stc_camera_protocol import STCFourSettingSource, load_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--plan-cache', required=True)
    parser.add_argument('--planner-json', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    inputs = [Path(args.dataroot).resolve(), Path(args.plan_cache).resolve(), Path(args.planner_json).resolve()]
    if any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):
        raise ValueError('audit output must not contain or sit inside input data/cache/JSON')
    if out.exists():
        raise FileExistsError('new output directory required; never overwrite an old audit')
    started = time.perf_counter()
    manifest_path = Path(args.plan_cache) / 'manifest.json'
    digest = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    scenes = {key.split('__')[0] for key in manifest.get('keys', [])}
    if not scenes:
        raise ValueError('empty planner manifest')
    print('Loading only nuScenes identity/ego metadata; no sensor measurements/labels/model.', flush=True)
    catalog, provenance = load_catalog(args.dataroot, scenes)
    # This constructor validates window identity only. No preflight or image paths are opened.
    source = STCFourSettingSource(args.dataroot, args.plan_cache, args.plan_cache, catalog, cache_mib=0)
    if source.metadata['manifest_sha256'] != digest:
        raise RuntimeError('planner manifest changed while loading metadata')
    result = audit(source, args.planner_json,
        progress=lambda i, n: print(f'PLANNER_ORIGIN {i}/{n}', flush=True))
    for record in provenance:
        if sha256(record['path']) != record['sha256']:
            raise RuntimeError('nuScenes metadata changed during audit')
    result['metadata_files'] = provenance
    result['elapsed_seconds'] = time.perf_counter() - started
    out.mkdir(parents=True, exist_ok=False)
    (out / 'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    text = summary(result) + f"seconds={result['elapsed_seconds']:.2f}\n"
    (out / 'summary.txt').write_text(text, encoding='utf-8')
    print(text, end='')
    print('Report: ' + str(out / 'summary.txt'))


if __name__ == '__main__':
    main()
