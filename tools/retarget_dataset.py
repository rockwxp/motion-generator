"""Process only an explicit screened selection; preserve every existing outcome."""
import argparse
from collections import Counter
import fcntl
import json
import os
from time import perf_counter
from pathlib import Path

from check import ROOT, DEFAULT_ROBOT
from download_dataset import file_hash, FOLDER
from retarget import retarget_batch


def read_records(path):
    records = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f'Incomplete record requires inspection: {path}')
            records[row['source']] = row
    return records


def paths(source):
    relative = source.relative_to(ROOT/'motions/source')
    return (ROOT/'motions/g1_23dof'/relative.with_suffix('.npz'),
            DEFAULT_ROBOT/'verification'/relative.with_suffix('.json'))



def existing_comparisons(source):
    """Find isolated results that must be reviewed before any production rerun."""
    relative = source.relative_to(ROOT/'motions/source')
    reports = (DEFAULT_ROBOT/'verification/comparisons').glob('*/'+relative.with_suffix('.json').as_posix())
    outputs = (ROOT/'motions/g1_23dof/comparisons').glob('*/'+relative.with_suffix('.npz').as_posix())
    return sorted([*reports, *outputs])


def outcome(source):
    output, report = paths(source)
    row = {'source': str(source.relative_to(ROOT)), 'source_sha256': file_hash(source)}
    if not report.exists():
        raise ValueError('Existing output has no report; refusing regeneration.')
    checked = json.loads(report.read_text())
    if checked['provenance']['source_sha256'] != row['source_sha256']:
        raise ValueError('Existing report source mismatch.')
    if checked['passed'] and not output.exists():
        raise ValueError('Passing report has no output; refusing regeneration.')
    if not checked['passed'] and output.exists():
        raise ValueError('Output contradicts rejection report.')
    row.update(status='reference' if output.exists() else 'rejected',
               report=str(report.relative_to(ROOT)), report_sha256=file_hash(report),
               warnings=checked.get('warnings', []), failures=checked.get('failures', []),
               motion_fit='unverified', physics='unverified')
    if output.exists():
        row.update(output=str(output.relative_to(ROOT)), output_sha256=file_hash(output))
        velocity = checked.get('joint_velocity', {})
        if velocity.get('exceeding_joints'):
            row.update(status='skipped_velocity', reason='Reference speed exceeds declared URDF joint limits.')
        elif checked.get('warnings'):
            row.update(status='review', reason='Diagnostic warnings retained for review, not proof the motion is impossible.')
    return row



def validate_configuration():
    """Gate on verified software/configuration, not success of every individual motion."""
    review_path = DEFAULT_ROBOT/'verification/official_pipeline_review.json'
    if not review_path.exists():
        raise ValueError('Official pipeline verification is missing; bulk generation is disabled.')
    review = json.loads(review_path.read_text())
    if not review.get('official_pipeline_reproduced') or not review.get('native_configuration_files_unchanged_since_static_alignment_test'):
        raise ValueError('Official/native configuration verification failed; bulk generation is disabled.')
    static_path = DEFAULT_ROBOT/'verification/retargeting.json'
    if file_hash(static_path) != review['native_static_alignment_report_sha256']:
        raise ValueError('Static configuration verification record changed.')
    for name, digest in json.loads(static_path.read_text())['config_sha256'].items():
        if file_hash(DEFAULT_ROBOT/name) != digest:
            raise ValueError(f'Configuration changed since verification: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, help='Screening decisions JSONL; only keep rows are used.')
    parser.add_argument('--limit', type=int, help='Limit pending entries for representative validation.')
    parser.add_argument('--batch-size', type=int, default=1, help='Maximum pending BVHs per SOMA batch.')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--reuse-only', action='store_true', help='Account for existing outcomes only; never run IK.')
    args = parser.parse_args()
    ledger = ROOT/'motions/g1_23dof/selected_library.jsonl'
    if args.status:
        print(json.dumps(dict(Counter(x['status'] for x in read_records(ledger).values())), indent=2))
        return
    if args.selection is None:
        parser.error('An explicit --selection is required. Full-dataset processing is not supported.')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')
    rows = [json.loads(line) for line in args.selection.read_text().splitlines() if line.strip()]
    selected = [x['source'] for x in rows if x['decision'] == 'keep']
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Selection is empty or contains duplicates.')
    approved_path = FOLDER/'screening/expanded_upper_body/decisions.jsonl'
    approved = {row['source'] for row in
                (json.loads(line) for line in approved_path.read_text().splitlines() if line.strip())
                if row['decision'] == 'keep'}
    if not set(selected).issubset(approved):
        raise ValueError('Selection includes sources outside the approved screened candidates.')
    index = {x['source']: x['sha256'] for x in
             (json.loads(line) for line in (FOLDER/'dataset_index.jsonl').read_text().splitlines())}
    for name in selected:
        source = (ROOT/name).resolve()
        source.relative_to(ROOT/'motions/source')
        if name not in index:
            raise ValueError(f'Source outside verified dataset: {name}')
    if not args.audit and not args.reuse_only and args.limit is None:
        representatives = {json.loads(line)['source'] for line in
                           (FOLDER/'representatives.jsonl').read_text().splitlines() if line.strip()}
        if not set(selected).issubset(representatives):
            validate_configuration()
    with (ROOT/'motions/g1_23dof/.library.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = read_records(ledger)
        legacy = read_records(ROOT/'motions/g1_23dof/library.jsonl')
        processed = 0
        pending = []

        def record(row, reused):
            nonlocal processed
            name = row['source']
            row['reused'] = reused
            with ledger.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            done[name] = row
            processed += 1
            print(json.dumps({'processed': processed, 'source': name, 'status': row['status'], 'reused': reused}), flush=True)

        def flush_batch():
            if not pending:
                return
            start = perf_counter()
            for source, error in retarget_batch(pending):
                if error is None:
                    row = outcome(source)
                else:
                    name = str(source.relative_to(ROOT))
                    row = {'source': name, 'source_sha256': index[name], 'status': 'skipped_error',
                           'reason': error, 'motion_fit': 'unverified', 'physics': 'unverified',
                           'preserved_files_sha256': {str(p.relative_to(ROOT)): file_hash(p)
                                                      for p in paths(source) if p.exists()}}
                record(row, False)
            print(json.dumps({'batch_clips': len(pending), 'batch_seconds': perf_counter() - start}), flush=True)
            pending.clear()

        for name in selected:
            source = ROOT/name
            if file_hash(source) != index[name]:
                raise ValueError(f'Changed source: {name}')
            if name in done:
                if done[name]['source_sha256'] != index[name]:
                    raise ValueError(f'Saved source hash mismatch: {name}')
                for field in ['output', 'report']:
                    if field in done[name] and file_hash(ROOT/done[name][field]) != done[name][field+'_sha256']:
                        raise ValueError(f'Changed saved result: {name}')
                for path, digest in done[name].get('preserved_files_sha256', {}).items():
                    if file_hash(ROOT/path) != digest:
                        raise ValueError(f'Changed preserved comparison: {path}')
                continue
            if args.audit:
                raise ValueError(f'Unprocessed selected source: {name}')
            if args.limit is not None and processed + len(pending) >= args.limit:
                break
            output, report = paths(source)
            reused = output.exists() or report.exists()
            if reused:
                if name in legacy:
                    for field in ['output', 'report']:
                        if field in legacy[name] and file_hash(ROOT/legacy[name][field]) != legacy[name][field+'_sha256']:
                            raise ValueError(f'Changed legacy result: {name}')
            comparisons = existing_comparisons(source) if not reused else []
            if comparisons:
                row = {'source': name, 'source_sha256': index[name], 'status': 'skipped_review',
                       'reason': 'Existing comparison results preserved; no production rerun or automatic promotion.',
                       'motion_fit': 'unverified', 'physics': 'unverified',
                       'preserved_files_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in comparisons}}
                for p in comparisons:
                    if p.suffix == '.json' and json.loads(p.read_text())['provenance']['source_sha256'] != index[name]:
                        raise ValueError(f'Comparison source mismatch: {p}')
                reused = True
                record(row, True)
            elif reused:
                record(outcome(source), True)
            elif not args.reuse_only:
                pending.append(source)
                if len(pending) == args.batch_size:
                    flush_batch()
        flush_batch()
        print(json.dumps({'selected': len(selected), 'accounted': sum(x in done for x in selected),
                          'counts': dict(Counter(done[x]['status'] for x in selected if x in done))}), flush=True)


if __name__ == '__main__':
    main()
