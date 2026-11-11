"""Check a robot trajectory with MuJoCo forward kinematics; never step physics."""
import argparse
from functools import lru_cache
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT = ROOT / 'robots/g1_23dof'


@lru_cache(maxsize=4)
def _load_model(xml_path, modified_ns):
    # One immutable model per path/version; every check still gets fresh MjData.
    return mujoco.MjModel.from_xml_path(xml_path)


def model_and_names(robot):
    robot = Path(robot)
    manifest = json.loads((robot / 'manifest.json').read_text())
    xml = robot / manifest['desc']['xml_path']
    model = _load_model(str(xml.resolve()), xml.stat().st_mtime_ns)
    joints = model.actuator_trnid[:, 0]
    if len(set(joints.tolist())) != model.nu or not np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError('Expected unique actuated hinge joints.')
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in joints]
    return model, joints, names



def joint_speed_report(motion, robot=DEFAULT_ROBOT):
    """Compare finite-difference speeds with the matching official URDF limits; no FK."""
    robot = Path(robot)
    model, joints, names = model_and_names(robot)
    manifest = json.loads((robot / 'manifest.json').read_text())
    urdf = (robot / manifest['desc']['xml_path']).with_suffix('.urdf')
    raw = urdf.read_bytes()
    nodes = {node.get('name'): node.find('limit') for node in ET.fromstring(raw).findall('joint')}
    limits = []
    for index, name in enumerate(names):
        node = nodes.get(name)
        if node is None:
            raise ValueError(f'Missing URDF joint limit: {name}')
        bounds = [float(node.get('lower')), float(node.get('upper'))]
        # The official MJCF rounds some bounds to five decimal places.
        if not np.allclose(bounds, model.jnt_range[joints[index]], atol=5e-6, rtol=0):
            raise ValueError(f'URDF and MJCF position limits differ: {name}')
        limits.append(float(node.get('velocity')))
    limits = np.asarray(limits)
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError('URDF velocity limits must be finite and positive.')
    times = np.asarray(motion['timestamps'], dtype=float)
    positions = np.asarray(motion['joint_pos'], dtype=float)
    if (np.asarray(motion['joint_names']).tolist() != names or times.ndim != 1
            or len(times) < 2 or positions.shape != (len(times), len(names))
            or not np.isfinite(times).all() or not np.isfinite(positions).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid trajectory timing, positions or joint order.')
    speed = np.abs(np.diff(positions, axis=0)) / np.diff(times)[:, None]
    peaks = np.argmax(speed, axis=0)
    return {
        'source': str(urdf.relative_to(robot)), 'urdf_sha256': hashlib.sha256(raw).hexdigest(),
        'scope': 'Declared URDF velocity bounds only; not acceleration, torque or balance validation.',
        'exceeding_joints': [name for index, name in enumerate(names) if speed[peaks[index], index] > limits[index]],
        'joints': {name: {'max_rad_s': float(speed[peaks[index], index]),
                          'limit_rad_s': float(limits[index]),
                          'max_to_limit_ratio': float(speed[peaks[index], index] / limits[index]),
                          'frame_interval': [int(peaks[index]), int(peaks[index] + 1)]}
                   for index, name in enumerate(names)}}


def check_motion(motion, robot=DEFAULT_ROBOT, *, targets=None, align_ground=False):
    """Return JSON-compatible metrics and failures. Limits are screening settings."""
    robot = Path(robot)
    cfg = json.loads((robot / 'retarget/checks.json').read_text())
    model, joints, names = model_and_names(robot)
    result = {'passed': False, 'failures': [], 'warnings': [], 'metrics': {}, 'thresholds': cfg,
              'acceptance_scope': 'Data format and model joint limits only; warnings and target fit require review.',
              'scope': 'Kinematic screening only; no dynamics or hardware validation.'}
    fail = result['failures'].append
    warn = result['warnings'].append
    required = ['timestamps', 'root_pos', 'root_quat', 'joint_pos', 'joint_names']
    if any(k not in motion for k in required):
        fail('Missing required fields.'); return result
    t, p, q, j = [np.asarray(motion[k]) for k in required[:4]]
    n = len(t) if t.ndim == 1 else 0
    if n < 2 or p.shape != (n, 3) or q.shape != (n, 4) or j.shape != (n, model.nu):
        fail('Invalid array shapes or fewer than two frames.'); return result
    if np.asarray(motion['joint_names']).tolist() != names:
        fail('Joint names/order do not match model actuators.'); return result
    if not all(np.isfinite(a).all() for a in [t, p, q, j]):
        fail('Non-finite trajectory values.'); return result
    dt = np.diff(t)
    if np.any(dt <= 0):
        fail('Timestamps must increase strictly.'); return result
    if np.max(np.abs(np.linalg.norm(q, axis=1) - 1)) > cfg['quaternion_norm_tolerance']:
        fail('Root quaternions are not unit length.'); return result
    if model.nq != 7 + model.nu or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Expected one free root and independently actuated hinge joints.')
    low, high = model.jnt_range[joints].T
    excess = float(max(0, np.max(low - j), np.max(j - high)))
    saturation = np.mean((j - low < cfg['limit_margin_rad']) | (high - j < cfg['limit_margin_rad']), axis=0)
    result['joint_velocity'] = joint_speed_report(motion, robot)
    speed = max(item['max_rad_s'] for item in result['joint_velocity']['joints'].values())
    if result['joint_velocity']['exceeding_joints']:
        warn('Declared URDF velocity exceeded: ' + ', '.join(result['joint_velocity']['exceeding_joints']))
    root_speed = float(np.max(np.linalg.norm(np.diff(p, axis=0), axis=1) / dt))
    # Relative quaternion via chord distance is stable for very small rotations.
    qa, qb = q[:-1].astype(float), q[1:].astype(float)
    qa /= np.linalg.norm(qa, axis=1)[:, None]; qb /= np.linalg.norm(qb, axis=1)[:, None]
    chord = np.minimum(np.linalg.norm(qb-qa, axis=1), np.linalg.norm(qb+qa, axis=1))
    root_spin = float(np.max(4 * np.arcsin(np.clip(chord / 2, 0, 1)) / dt))
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in cfg['foot_bodies']]
    if min(feet) < 0:
        raise ValueError('Configured foot body is missing.')
    spheres = [np.flatnonzero((model.geom_bodyid == body) & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
                             & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))) for body in feet]
    if any(len(ids) == 0 for ids in spheres):
        raise ValueError('Foot screening requires the model foot contact spheres.')
    target_ids = []
    if targets is not None:
        tx = np.asarray(targets['transforms'])
        target_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in targets['bodies']]
        if tx.shape != (n, len(target_ids), 7) or not np.isfinite(tx).all() or min(target_ids) < 0:
            raise ValueError('Invalid target transforms or missing target body.')
        if not np.allclose(np.linalg.norm(tx[..., 3:], axis=-1), 1, atol=1e-4):
            raise ValueError('Target quaternions must be normalized xyzw.')
        pos_error = np.empty((n, len(target_ids)))
        rot_error = np.empty_like(pos_error)
    data = mujoco.MjData(model)
    foot_xyz = np.empty((n, len(feet), 3)); heights = np.empty((n, len(feet)))
    max_collision = 0.; worst_pair = None; worst_frame = None
    for frame in range(n):
        data.qpos[:3] = p[frame]; data.qpos[3:7] = q[frame]
        data.qpos[model.jnt_qposadr[joints]] = j[frame]
        mujoco.mj_forward(model, data)
        for k, body in enumerate(target_ids):
            pos_error[frame, k] = np.linalg.norm(data.xpos[body] - tx[frame, k, :3])
            target_q = tx[frame, k, [6, 3, 4, 5]]
            target_q = target_q / np.linalg.norm(target_q)
            dot = abs(float(np.dot(data.xquat[body], target_q)))
            rot_error[frame, k] = 2 * np.arccos(np.clip(dot, 0, 1))
        for f, ids in enumerate(spheres):
            foot_xyz[frame, f] = np.mean(data.geom_xpos[ids], axis=0)
            heights[frame, f] = np.min(data.geom_xpos[ids, 2] - model.geom_size[ids, 0])
        for contact in data.contact[:data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if (model.geom_bodyid[a] > 0 and model.geom_bodyid[b] > 0
                    and model.geom_bodyid[a] != model.geom_bodyid[b] and -contact.dist > max_collision):
                max_collision = float(-contact.dist); worst_frame = frame
                worst_pair = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) for g in [a,b]]
    raw_min = float(heights.min())
    shift = 0.0
    if align_ground:
        if abs(raw_min) <= cfg['ground_alignment_max_shift_m']:
            shift = -raw_min
            motion['root_pos'] = p.copy()
            motion['root_pos'][:, 2] += shift
            heights += shift
            foot_xyz[:, :, 2] += shift
        else:
            warn('Ground alignment exceeds the legacy diagnostic bound; no shift applied.')
        result['ground_alignment'] = {'constant_z_translation_m': shift, 'raw_min_foot_height_m': raw_min}
    if targets is not None:
        def summarize(values):
            return {'mean': float(values.mean()), 'p95': float(np.percentile(values, 95)),
                    'max': float(values.max()), 'worst_frame': int(values.argmax())}
        result['target_errors'] = {
            'reference': 'SOMA scaled and offset source targets, before contact post-processing; both target and robot share any constant ground shift.',
            'interpretation': 'Task expression error, not post-processing solver residual or a waist-DOF necessity test. Original-prefixed effectors retain production-config targets when testing another scaler. No acceptance threshold.',
            'effectors': {name: {'body': targets['bodies'][k], 'position_m': summarize(pos_error[:, k]),
                               'rotation_rad': summarize(rot_error[:, k])}
                          for k, name in enumerate(targets['names'])}}
    velocity = np.diff(foot_xyz, axis=0) / dt[:, None, None]
    near_ground = (heights[:-1] <= cfg['contact_height_m']) & (heights[1:] <= cfg['contact_height_m'])
    stance = near_ground & (np.abs(velocity[:, :, 2]) <= cfg['contact_vertical_speed_m_s'])
    contact_method = 'Robot height and vertical speed heuristic'
    if 'source_foot_contact' in motion:
        contact = np.asarray(motion['source_foot_contact'])
        if (contact.shape != (n, len(feet)) or not np.isfinite(contact).all()
                or np.any(contact < 0) or np.any(contact > 1)):
            fail('Invalid source contact array.'); return result
        stance = (contact[:-1] >= 0.999) & (contact[1:] >= 0.999)
        contact_method = 'SOMA source contact detector; consecutive full-contact frames'
    horizontal = np.linalg.norm(velocity[:, :, :2], axis=2)
    slide = float(np.max(horizontal[stance])) if np.any(stance) else None
    penetration = float(max(0, -heights.min()))
    grounded = float(np.mean(np.any(heights <= cfg['contact_height_m'], axis=1)))
    metrics = {'frames': n, 'duration_s': float(t[-1]-t[0]), 'fps': float(1/np.median(dt)),
               'joint_limit_excess_rad': excess, 'max_joint_speed_rad_s': speed,
               'max_root_speed_m_s': root_speed, 'max_root_angular_speed_rad_s': root_spin,
               'max_joint_limit_saturation_fraction': float(saturation.max()),
               'min_foot_height_m': float(heights.min()), 'max_foot_penetration_m': penetration, 'max_stance_foot_speed_m_s': slide,
               'stance_intervals': int(stance.sum()), 'contact_method': contact_method, 'grounded_frame_fraction': grounded,
               'max_self_penetration_m': max_collision, 'worst_collision_pair': worst_pair,
               'worst_collision_frame': worst_frame}
    # Reuse the existing FK heights: distinguish a world-height offset from variation.
    def height_distribution(values):
        if values.size == 0:
            return None
        return dict(zip(('min', 'p05', 'median', 'p95', 'max'),
                        np.percentile(values, [0, 5, 50, 95, 100]).tolist()))
    support_height = heights.min(axis=1)
    result['ground_profile'] = {
        'scope': 'Foot contact-sphere bottoms in the saved world frame; diagnostic only, no ground correction.',
        'lowest_foot_height_m': height_distribution(support_height),
        'worst_frame': int(support_height.argmin()),
        'constant_shift_to_clear_all_feet_m': float(-support_height.min()),
        'contact_foot_height_m': {name: height_distribution(heights[:-1, index][stance[:, index]])
                                  for index, name in enumerate(cfg['foot_bodies'])},
        'contact_method': contact_method}
    result['metrics'] = metrics
    for metric, limit in [('joint_limit_excess_rad','joint_limit_tolerance_rad'),
                          ('max_joint_speed_rad_s','max_joint_speed_rad_s'),
                          ('max_root_speed_m_s','max_root_speed_m_s'),
                          ('max_root_angular_speed_rad_s','max_root_angular_speed_rad_s'),
                          ('max_joint_limit_saturation_fraction','max_limit_saturation_fraction'),
                          ('max_foot_penetration_m','max_foot_penetration_m'),
                          ('max_self_penetration_m','max_self_penetration_m')]:
        if metrics[metric] > cfg[limit]:
            (fail if metric == 'joint_limit_excess_rad' else warn)(f'{metric}: {metrics[metric]:.6g} > {cfg[limit]:.6g}')
    if slide is None: warn('No stance intervals found; foot-slide check is inconclusive.')
    elif slide > cfg['max_stance_foot_speed_m_s']: warn(f'Stance foot speed: {slide:.6g} m/s exceeds limit.')
    if grounded < cfg['min_grounded_frame_fraction']: warn('Insufficient ground contact for the flat-ground task.')
    result['passed'] = not result['failures']
    result['review_required'] = True
    result['motion_fit_status'] = 'Requires representative review; target metrics are diagnostic, not acceptance thresholds.'
    return result


def review_trajectory(trajectory, source=None, robot=DEFAULT_ROBOT):
    """Review an existing trajectory without changing it or rerunning IK."""
    with np.load(trajectory, allow_pickle=False) as archive:
        motion = {name: archive[name].copy() for name in archive.files}
    targets = None
    if source is not None:
        import hashlib
        import warp as wp
        from retarget import load_source, source_targets
        source = Path(source)
        metadata = json.loads(str(motion['metadata_json']))
        if hashlib.sha256(source.read_bytes()).hexdigest() != metadata['source_sha256']:
            raise ValueError('Source does not match the saved trajectory.')
        for name, digest in metadata['files_sha256'].items():
            if hashlib.sha256((Path(robot)/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Robot settings differ from saved motion: {name}')
        wp.init()
        with wp.ScopedDevice('cpu'):
            skeleton, animation, _ = load_source(source)
            expected = np.arange(animation.num_frames)/animation.sample_rate
            if motion['timestamps'].shape != expected.shape or not np.allclose(motion['timestamps'], expected, atol=1e-7, rtol=0):
                raise ValueError('Source and trajectory timing differ.')
            targets = source_targets(animation, robot, skeleton=skeleton, config=metadata.get('experimental_retarget_config'))
        saved_report = json.loads(str(motion['check_report_json']))
        shift = saved_report.get('ground_alignment', {}).get('constant_z_translation_m', 0.0)
        targets['transforms'][:, :, 2] += shift
    report = check_motion(motion, robot, targets=targets)
    import hashlib
    report['review_provenance'] = {
        'trajectory_sha256': hashlib.sha256(Path(trajectory).read_bytes()).hexdigest(),
        'check_tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'target_tool_sha256': hashlib.sha256(Path(__file__).with_name('retarget.py').read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None,
        'retargeting_repeated': False}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--robot', type=Path, default=DEFAULT_ROBOT)
    parser.add_argument('--source', type=Path, help='Original BVH for SOMA target-error diagnostics; no IK.')
    parser.add_argument('--report', type=Path, help='Write a new supplementary report; never overwrite.')
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error('Report already exists; it will not be overwritten.')
    report = review_trajectory(args.trajectory, args.source, args.robot)
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as stream:
            stream.write(text)
    print(text)
    raise SystemExit(0 if report['passed'] else 1)
