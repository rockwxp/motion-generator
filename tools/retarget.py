"""Retarget one SOMA BVH, check it, and save format-valid reference trajectories with diagnostic reports."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import re
from pathlib import Path
import tempfile

import numpy as np
from check import ROOT, DEFAULT_ROBOT, check_motion, model_and_names


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(source):
    from soma_retargeter.io.bvh import load_bvh
    from soma_retargeter.pipelines import utils
    source = Path(source).resolve()
    if not source.is_file() or source.suffix.lower() != '.bvh':
        raise ValueError('Expected a BVH file.')
    skeleton, animation = load_bvh(str(source))
    frame_time_match = re.search(r'^Frame Time:\s*([0-9.eE+-]+)', source.read_text(), re.MULTILINE)
    if frame_time_match is None: raise ValueError('Missing BVH frame time.')
    frame_time = float(frame_time_match[1])
    if not np.isfinite(frame_time) or frame_time <= 0: raise ValueError('Invalid BVH frame time.')
    animation.sample_rate = 1.0 / frame_time  # Preserve the header instead of the loader's rounded rate.
    if animation.num_frames < 2 or not np.isfinite(animation.sample_rate) or animation.sample_rate <= 0:
        raise ValueError('Invalid frame count or sample rate.')
    if not np.isfinite(animation.local_transforms).all(): raise ValueError('Non-finite source transforms.')
    reference, _ = load_bvh(utils.get_source_zero_pose_asset_path(utils.SourceType.SOMA))
    if skeleton.joint_names != reference.joint_names or list(skeleton.parent_indices) != list(reference.parent_indices):
        raise ValueError('Input does not match the SOMA reference skeleton topology.')
    # Official samples share SOMA topology but may have small rest-offset differences.
    # Preserve their offsets, as the upstream retargeter does, and report the difference.
    rest_offset_difference = float(np.max(np.linalg.norm(
        skeleton.reference_local_transforms[1:, :3] - reference.reference_local_transforms[1:, :3], axis=1)))
    return skeleton, animation, rest_offset_difference


def save_result(source, robot, animation, rest_offset_difference, pipeline, buffer, contact_index=0, *, experiment=None, config=None):
    source, robot = Path(source).resolve(), Path(robot).resolve()
    source_relative = source.relative_to(ROOT / 'motions/source')
    destination = source_relative if experiment is None else Path('comparisons')/experiment/source_relative
    output = ROOT / 'motions' / robot.name / destination.with_suffix('.npz')
    report_path = robot / 'verification' / destination.with_suffix('.json')
    if output.exists() or report_path.exists():
        raise FileExistsError('Existing trajectory or report is preserved; use check.py to review it.')
    manifest = json.loads((robot / 'manifest.json').read_text())
    raw = np.asarray(buffer.data)
    model, joints, names = model_and_names(robot)
    newton_names = [name.split('/')[-1] for name in pipeline.robot_builder.joint_label][1:]
    if set(newton_names) != set(names) or raw.shape != (animation.num_frames, 7+len(names)):
        raise ValueError('Unexpected solver output dimensions or joints.')
    if not np.isclose(buffer.sample_rate, animation.sample_rate): raise ValueError('Solver changed source timing.')
    q = raw[:, [6,3,4,5]].copy()
    for i in range(1,len(q)):
        if np.dot(q[i-1], q[i]) < 0: q[i] *= -1
    motion = {'timestamps': np.arange(len(raw), dtype=np.float64) / animation.sample_rate,
              'root_pos': raw[:, :3], 'root_quat': q,
              'joint_pos': raw[:, [7+newton_names.index(name) for name in names]],
              'joint_names': np.asarray(names)}
    if pipeline.contact_results:
        channels = [pipeline._contact_joint_names.index(n) for n in ['LeftFoot', 'RightFoot']]
        motion['source_foot_contact'] = pipeline.contact_results[contact_index].contacts[:, channels]
    targets = source_targets(animation, robot, skeleton=animation.skeleton, config=config)
    report = check_motion(motion, robot, targets=targets, align_ground=True)
    if hasattr(pipeline, 'clamp_diagnostics'):
        report['soma_output_clamp'] = pipeline.clamp_diagnostics[contact_index]
    paths = ['manifest.json', manifest['desc']['xml_path'], 'retarget/retargeter.json',
             'retarget/scaler.json', 'retarget/post_processing.json', 'retarget/checks.json']
    if config is not None:
        for extra in [config['human_robot_scaler_config'], config.get('post_processing', {}).get('robot_config')]:
            if extra and extra not in paths:
                paths.append(extra)
    provenance = {'source': str(source_relative), 'max_source_rest_offset_difference_m': rest_offset_difference, 'source_sha256': sha256(source),
                  'robot': robot.name, 'files_sha256': {p:sha256(robot/p) for p in paths},
                  'soma_commit': json.loads(metadata.distribution('soma_retargeter').read_text('direct_url.json'))['vcs_info']['commit_id'],
                  'versions': {k:metadata.version(k) for k in ['soma_retargeter','newton','warp-lang','mujoco']},
                  'tool_sha256': {p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('check.py')]}}
    if config is not None:
        provenance['experimental_retarget_config'] = config
        provenance['experiment'] = experiment
    sidecar = source.with_suffix('.source.json')
    if sidecar.exists(): provenance['source_metadata'] = json.loads(sidecar.read_text())
    report['provenance'] = provenance
    report['output'] = str(output.relative_to(ROOT)) if report['passed'] else None
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report['passed']: return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    motion['metadata_json'] = np.asarray(json.dumps(provenance))
    motion['check_report_json'] = np.asarray(json.dumps(report))
    # Write in the destination filesystem; read back before publishing the file.
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.npz', delete=False) as f:
            temp = Path(f.name); np.savez_compressed(f, **motion)
        with np.load(temp, allow_pickle=False) as saved:
            for key, value in motion.items():
                if not np.array_equal(saved[key], value): raise ValueError(f'Save/read-back mismatch: {key}')
        temp.replace(output)
    finally:
        if temp is not None: temp.unlink(missing_ok=True)
    print(f'SAVED {output}')
    return 0



def adapt_arm_lengths(animation, factors):
    """Copy a SOMA buffer, scaling only four arm-bone translation vectors."""
    from soma_retargeter.animation.skeleton import Skeleton
    from soma_retargeter.animation.animation_buffer import AnimationBuffer
    if not factors:
        return animation
    expected = {side+part: side+parent for side in ('Left', 'Right')
                for part, parent in [('ForeArm', 'Arm'), ('Hand', 'ForeArm')]}
    if set(factors) != set(expected):
        raise ValueError('Arm adaptation requires exactly the four forearm/hand bone scales.')
    original = animation.skeleton
    reference = original.reference_local_transforms.copy()
    local = animation.local_transforms.copy()
    for name, parent in expected.items():
        index = original.joint_index(name)
        factor = float(factors[name])
        if (index < 0 or original.joint_name(original.joint_parent(index)) != parent
                or not np.isfinite(factor) or factor <= 0):
            raise ValueError('Invalid arm topology or bone scale.')
        reference[index, :3] *= factor
        local[:, index, :3] *= factor
    skeleton = Skeleton(original.num_joints, original.joint_names, original.parent_indices, reference)
    return AnimationBuffer(skeleton, animation.num_frames, animation.sample_rate, local)


def source_targets(animation, robot=DEFAULT_ROBOT, *, skeleton=None, config=None):
    """Use SOMA's scaler directly; this does not run retargeting or contact correction."""
    import warp as wp
    from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    robot = Path(robot)
    cfg = config or json.loads((robot/'retarget/retargeter.json').read_text())
    adapted = adapt_arm_lengths(animation, cfg.get('source_arm_scale'))
    scaler = HumanToRobotScaler(adapted.skeleton, str(robot/cfg['human_robot_scaler_config']))
    transform = SpaceConverter(FacingDirectionType.MUJOCO).transform(wp.transform_identity())
    tx = scaler.compute_effectors_from_buffer(adapted, True, transform)
    names = ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot']
    indices = [scaler.effector_names().index(n) for n in names]
    result = {'names': names, 'bodies': [cfg['ik_match_table'][n]['t_body'] for n in names],
              'transforms': tx[:, indices].copy()}
    if config is not None and cfg['human_robot_scaler_config'] != 'retarget/scaler.json':
        original = source_targets(animation, robot, skeleton=skeleton)
        result['names'] += ['Original'+name for name in original['names']]
        result['bodies'] += original['bodies']
        result['transforms'] = np.concatenate([result['transforms'], original['transforms']], axis=1)
    return result


def observe_output_clamping(pipeline):
    """Observe upstream output clamping without changing the solver or its output."""
    original = pipeline.joint_limit_clamper.apply
    names = [name.split('/')[-1] for name in pipeline.robot_builder.joint_label][1:]
    pipeline.clamp_diagnostics = []

    def observed(buffer):
        before = buffer.numpy().copy()
        result = original(buffer)
        after = buffer.numpy()
        if before.shape[1] != 7 + len(names):
            raise ValueError('Unexpected diagnostic joint layout.')
        difference = np.abs(after[:, 7:] - before[:, 7:])
        raw_steps = np.abs(np.diff(before[:, 7:], axis=0))
        saved_steps = np.abs(np.diff(after[:, 7:], axis=0))
        frame, joint = np.unravel_index(saved_steps.argmax(), saved_steps.shape)
        pipeline.clamp_diagnostics.append({
            'changed_frames': int(np.any(difference > 0, axis=1).sum()),
            'max_coordinate_change_rad': float(difference.max()),
            'per_joint_max_change_rad': dict(zip(names, difference.max(axis=0).tolist())),
            'largest_saved_step': {'joint': names[joint], 'frame_interval': [int(frame), int(frame+1)],
                                  'before_clamp_rad': float(raw_steps[frame, joint]),
                                  'after_clamp_rad': float(saved_steps[frame, joint]),
                                  'raw_values_rad': before[frame:frame+2, 7+joint].tolist(),
                                  'saved_values_rad': after[frame:frame+2, 7+joint].tolist()},
            'scope': 'Observation of SOMA final limit clamp; no filter or solver modification.'})
        return result

    pipeline.joint_limit_clamper.apply = observed


def retarget_one(source, robot=DEFAULT_ROBOT, device='cuda:0', *, experiment=None, config=None):
    if config and config.get('source_arm_scale'):
        raise ValueError('Custom bone adaptation is historical only; new runs must use pure SOMA.')
    source, robot = Path(source).resolve(), Path(robot).resolve()
    relative = source.relative_to(ROOT/'motions/source')
    if (experiment is None) != (config is None):
        raise ValueError('Experimental configuration requires a separate output name.')
    if experiment is not None:
        if not re.fullmatch(r'[a-z0-9_]+', experiment):
            raise ValueError('Invalid experiment name.')
        scaler = (robot/config['human_robot_scaler_config']).resolve()
        scaler.relative_to((robot/'retarget').resolve())
        if not scaler.is_file():
            raise ValueError('Experimental scaler must exist under the robot retarget directory.')
        relative = Path('comparisons')/experiment/relative
    output = ROOT/'motions'/robot.name/relative.with_suffix('.npz')
    report_path = robot/'verification'/relative.with_suffix('.json')
    if output.exists() or report_path.exists():
        raise FileExistsError('Existing trajectory or report is preserved; no re-retargeting.')
    _, animation, difference = load_source(source)
    pipeline, buffers = solve_motions([animation], robot, device, config)
    return save_result(source, robot, animation, difference, pipeline, buffers[0], experiment=experiment, config=config)


def solve_motions(animations, robot, device, config=None):
    import warp as wp
    from soma_retargeter.robotics.robot_registry import register_robots_path
    from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    wp.init()
    wp.set_device(device)
    register_robots_path(str(robot))
    manifest = json.loads((robot/'manifest.json').read_text())
    pipeline = SomaRetargetingPipeline(animations[0].skeleton, 'soma', manifest['name'], retarget_config=config)
    if config and config.get('adapt_initialization'):
        from soma_retargeter.animation.animation_buffer import AnimationBuffer
        pose = pipeline.initialization_pose
        reference = AnimationBuffer(pose.skeleton, 1, animations[0].sample_rate,
                                    pose.get_local_transforms()[None].copy())
        reference = adapt_arm_lengths(reference, config.get('source_arm_scale'))
        pose.set_local_transforms(reference.local_transforms[0])
    transform = SpaceConverter(FacingDirectionType.MUJOCO).transform(wp.transform_identity())
    pipeline.add_input_motions(animations, [transform] * len(animations), True)
    if config and config.get('record_clamp_diagnostics'):
        observe_output_clamping(pipeline)
    buffers = pipeline.execute()
    if len(buffers) != len(animations):
        raise RuntimeError('SOMA returned an unexpected number of motions.')
    return pipeline, buffers


def retarget_batch(sources, robot=DEFAULT_ROBOT, device='cuda:0'):
    """Run compatible pending clips together and report source-specific errors."""
    groups = {}
    for source in sources:
        try:
            skeleton, animation, difference = load_source(source)
        except (ValueError, FloatingPointError) as error:
            yield source, f'{type(error).__name__}: {error}'
            continue
        # load_source already verifies topology. The scaler also shares the rest pose.
        key = skeleton.reference_local_transforms.tobytes()
        group = groups.setdefault(key, [])
        if group:
            # SOMA checks skeleton identity; only share an exactly matching rest pose.
            animation.skeleton = group[0][1].skeleton
        group.append((source, animation, difference))

    for clips in groups.values():
        yield from solve_group(clips, robot, device)


def solve_group(clips, robot, device):
    try:
        pipeline, buffers = solve_motions([animation for _, animation, _ in clips], robot, device)
    except (ValueError, FloatingPointError) as error:
        # Isolate bad input before any file from this group has been published.
        if len(clips) == 1:
            yield clips[0][0], f'{type(error).__name__}: {error}'
        else:
            print(f"Splitting {len(clips)} inputs after SOMA error: {error}", flush=True)
            mid = len(clips) // 2
            yield from solve_group(clips[:mid], robot, device)
            yield from solve_group(clips[mid:], robot, device)
        return
    for i, (source, animation, difference) in enumerate(clips):
        try:
            save_result(source, robot, animation, difference, pipeline, buffers[i], i)
        except (ValueError, FloatingPointError) as error:
            yield source, f'{type(error).__name__}: {error}'
        else:
            yield source, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--robot', type=Path, default=DEFAULT_ROBOT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', type=Path, help='Native SOMA JSON configuration; requires --experiment.')
    parser.add_argument('--experiment', help='Separate comparison output directory; never replaces the library.')
    args = parser.parse_args()
    if (args.config is None) != (args.experiment is None):
        parser.error('--config and --experiment must be supplied together.')
    config = json.loads(args.config.read_text()) if args.config else None
    return retarget_one(args.source, args.robot, args.device, experiment=args.experiment, config=config)


if __name__ == '__main__':
    raise SystemExit(main())
