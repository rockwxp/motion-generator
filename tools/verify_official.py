"""Verify pinned official SOMA BVH/G1 pairs using the unmodified bundled 29DoF configuration."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'cbdb2b494a2af82137bf9b6f15122d60167281c4'
CLIPS = ('Neutral_walk_forward_002__A057', 'wave_R_001__A428', 'item_pick_up_standing_R_001__A410')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(relative, destination):
    raw_url = f'https://raw.githubusercontent.com/NVIDIA/soma-retargeter/{COMMIT}/{relative}'
    with urllib.request.urlopen(raw_url, timeout=60) as response:
        raw = response.read()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
        expected = re.search(rb'oid sha256:([0-9a-f]+)', raw).group(1).decode()
        size = int(re.search(rb'size (\d+)', raw).group(1))
        if destination.exists():
            if sha(destination) != expected or destination.stat().st_size != size:
                raise ValueError(f'Existing official asset differs: {destination}')
            return
        url = f'https://media.githubusercontent.com/media/NVIDIA/soma-retargeter/{COMMIT}/{relative}'
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Official asset LFS verification failed.')
    if destination.exists():
        if destination.read_bytes() != raw:
            raise ValueError(f'Existing official asset differs: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(raw)


def main():
    installed = json.loads(metadata.distribution('soma_retargeter').read_text('direct_url.json'))['vcs_info']['commit_id']
    if installed != COMMIT:
        raise ValueError('Installed SOMA does not match the official asset revision.')
    import warp as wp
    from soma_retargeter.io.bvh import load_bvh
    from soma_retargeter.io.csv import load_csv, save_csv
    from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    wp.init(); wp.set_device('cuda:0')
    source_dir = ROOT/'motions/source/soma_official'
    reference_dir = ROOT/'motions/g1_29dof/official_reference'
    output_dir = ROOT/'motions/g1_29dof/official_validation'
    output_dir.mkdir(parents=True, exist_ok=True)
    fetch('assets/motions/LICENSE.txt', source_dir/'LICENSE.txt')
    rows = []
    for stem in CLIPS:
        source = source_dir/f'{stem}.bvh'; reference = reference_dir/f'{stem}.csv'
        fetch(f'assets/motions/bvh/{stem}.bvh', source)
        fetch(f'assets/motions/csv/unitree_g1/{stem}.csv', reference)
        output = output_dir/f'{stem}.csv'; report_path = output_dir/f'{stem}.json'
        if report_path.exists():
            row = json.loads(report_path.read_text())
            for key, path in [('source', source), ('reference', reference), ('output', output)]:
                if sha(path) != row[key+'_sha256']:
                    raise ValueError(f'Changed preserved result: {path}')
            rows.append(row); print('REUSED', stem, flush=True); continue
        if output.exists():
            raise ValueError('Output without completed report requires inspection; no rerun.')
        skeleton, animation = load_bvh(str(source))
        pipeline = SomaRetargetingPipeline(skeleton, 'soma', 'unitree_g1')
        transform = SpaceConverter(FacingDirectionType.MUJOCO).transform(wp.transform_identity())
        pipeline.add_input_motions([animation], [transform], True)
        buffer = pipeline.execute()[0]
        names = [name.split('/')[-1] for name in pipeline.robot_builder.joint_label][1:]
        expected_names = reference.read_text().splitlines()[0].split(',')[7:]
        if len(names) != 29 or expected_names != names:
            raise ValueError('Official reference joint order does not match the bundled 29DoF model.')
        actual = np.asarray(buffer.data, dtype=float)
        expected = np.asarray(load_csv(str(reference), animation.sample_rate).data, dtype=float)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError('Official frame count/shape mismatch or non-finite output.')
        qa = actual[:, 3:7]; qb = expected[:, 3:7]
        qa = qa/np.linalg.norm(qa, axis=1)[:, None]; qb = qb/np.linalg.norm(qb, axis=1)[:, None]
        angles = 2*np.arccos(np.clip(np.abs((qa*qb).sum(axis=1)), 0, 1))
        delta = np.abs(actual[:, 7:]-expected[:, 7:])
        save_csv(str(output), names, buffer)
        row = {'motion':stem, 'soma_commit':COMMIT, 'frames':len(actual), 'sample_rate':animation.sample_rate,
               'joint_count':len(names), 'source_sha256':sha(source), 'reference_sha256':sha(reference), 'output_sha256':sha(output),
               'max_root_position_difference_m':float(np.linalg.norm(actual[:,:3]-expected[:,:3], axis=1).max()),
               'max_root_rotation_difference_rad':float(angles.max()),
               'max_joint_difference_rad':float(delta.max()), 'mean_joint_difference_rad':float(delta.mean()),
               'worst_joint':names[int(np.unravel_index(delta.argmax(), delta.shape)[1])],
               'scope':'Direct official pipeline, unchanged bundled 29DoF config, no custom ground alignment or filtering. Numerical comparison only; not physics validation.'}
        with report_path.open('x') as stream: json.dump(row, stream, indent=2)
        rows.append(row); print(json.dumps(row), flush=True)
    summary = output_dir/'summary.json'; text = json.dumps(rows, indent=2)+'\n'
    if summary.exists():
        if summary.read_text() != text: raise ValueError('Existing summary differs.')
    else:
        with summary.open('x') as stream:stream.write(text)


if __name__ == '__main__':
    main()
