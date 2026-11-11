"""Replay a saved robot trajectory in MuJoCo without physics or hardware control."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from queue import SimpleQueue
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from check import ROOT, DEFAULT_ROBOT, model_and_names


def load_trajectory(path, names):
    with np.load(path, allow_pickle=False) as archive:
        times = archive['timestamps'].astype(np.float64)
        positions = archive['root_pos']
        rotations = archive['root_quat']
        joints = archive['joint_pos']
        if archive['joint_names'].tolist() != names:
            raise ValueError('Trajectory joint names/order do not match the robot model.')
    n = len(times) if times.ndim == 1 else 0
    if n < 2 or positions.shape != (n, 3) or rotations.shape != (n, 4) or joints.shape != (n, len(names)):
        raise ValueError('Invalid trajectory dimensions.')
    if not all(np.isfinite(a).all() for a in [times, positions, rotations, joints]):
        raise ValueError('Non-finite trajectory values.')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must increase strictly.')
    if not np.allclose(np.linalg.norm(rotations, axis=1), 1.0, atol=1e-4, rtol=0):
        raise ValueError('Root quaternions must be normalized wxyz values.')
    return times - times[0], positions, rotations, joints



def load_source(path, trajectory, times, robot_positions):
    """Load the original BVH in the same world axes used by retarget.py."""
    import warp as wp
    from soma_retargeter.io.bvh import load_bvh
    from soma_retargeter.utils.pose_utils import compute_global_poses_batch
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType

    raw = path.read_bytes()
    with np.load(trajectory, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
    if hashlib.sha256(raw).hexdigest() != metadata['source_sha256']:
        raise ValueError('BVH does not match the source recorded in the trajectory.')
    match = re.search(r'Frame Time:\s*([0-9.eE+-]+)', raw.decode('utf-8'))
    if match is None:
        raise ValueError('BVH has no frame time.')
    dt = float(match.group(1))
    wp.init()
    with wp.ScopedDevice('cpu'):
        skeleton, animation = load_bvh(str(path))
        if animation.num_frames != len(times) or not np.allclose(
            np.arange(animation.num_frames) * dt, times, atol=1e-7, rtol=0
        ):
            raise ValueError('BVH and trajectory must have identical frames and timing.')
        transform = SpaceConverter(FacingDirectionType.MUJOCO).transform(wp.transform_identity())
        points = compute_global_poses_batch(skeleton, animation.local_transforms, transform)[..., :3].copy()
    if not np.isfinite(points).all():
        raise ValueError('BVH contains non-finite joint positions.')
    parents = np.asarray(skeleton.parent_indices).copy()
    if skeleton.joint_names[:2] == ['Root', 'Hips'] and parents[1] == 0:
        # Root is a BVH reference node, not a human bone. Display from Hips.
        points = points[:, 1:]
        parents = parents[1:] - 1
    # Constant display translation only: preserve height, scale and motion.
    points[..., :2] += robot_positions[0, :2] + [0, 1.5] - points[0, 0, :2]
    return points, parents


def draw_source(scene, points, parents, robot_position):
    """Append a human skeleton and labels to a MuJoCo scene."""
    if scene.ngeom + 2 * len(parents) + 2 > scene.maxgeom:
        raise ValueError('BVH skeleton exceeds viewer geometry capacity.')

    def geom(kind, position, size, color):
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, kind, np.array(size), np.array(position), np.eye(3).ravel(), np.array(color, dtype=np.float32))
        scene.ngeom += 1
        return g

    for joint, parent in enumerate(parents):
        geom(mujoco.mjtGeom.mjGEOM_SPHERE, points[joint], [.009, 0, 0], [.1, .7, 1, 1])
        if parent >= 0 and np.linalg.norm(points[joint] - points[parent]) > 1e-8:
            g = geom(mujoco.mjtGeom.mjGEOM_CAPSULE, points[joint], [.006, .01, 0], [.1, .7, 1, 1])
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, .006, points[parent], points[joint])
    for label, position in [('Source BVH', [*points[0, :2], points[:, 2].max() + .15]),
                            ('G1 23DoF', robot_position + [0, 0, .75])]:
        g = geom(mujoco.mjtGeom.mjGEOM_LABEL, position, [0, 0, 0], [1, 1, 1, 1])
        g.label = label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('trajectory', nargs='?', type=Path)
    inputs.add_argument('--all', action='store_true', help='Play the training list once with matching source BVHs.')
    parser.add_argument('--source', type=Path, help='Original BVH for synchronized side-by-side replay.')
    parser.add_argument('--robot', type=Path, default=DEFAULT_ROBOT)
    parser.add_argument('--speed', type=float, default=1.0, help='Playback rate; 0.5 is half speed.')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive and finite')
    if args.all and args.source:
        parser.error('--all automatically loads the matching BVH for each trajectory')
    model, joint_ids, names = model_and_names(args.robot)
    if model.nq != 7 + len(names) or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Expected one free root and independent actuated joints.')
    if args.all:
        playlist = ROOT / 'robots/g1_23dof/verification/legacy_recheck/training_paths.txt'
        paths = [ROOT / line for line in playlist.read_text().splitlines() if line.strip()]
        if not paths:
            parser.error('Training playlist is empty.')
    else:
        paths = [args.trajectory]
    keys = SimpleQueue()
    data = mujoco.MjData(model)
    addresses = model.jnt_qposadr[joint_ids]
    threads_before_launch = set(threading.enumerate())
    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put) as viewer:
        viewer_threads = set(threading.enumerate()) - threads_before_launch
        for number, path in enumerate(paths, start=1):
            if not viewer.is_running():
                break
            source_path = args.source
            if args.all:
                relative = path.relative_to(ROOT / 'motions/g1_23dof')
                source_path = ROOT / 'motions/source' / relative.with_suffix('.bvh')
            print(f'[{number}/{len(paths)}] {path.stem}', flush=True)
            play_clip(viewer, keys, model, data, addresses, names, path, source_path, args.speed, args.all)

    # The Linux passive viewer releases its OpenGL context on a background thread.
    for thread in viewer_threads:
        thread.join(timeout=5)


def play_clip(viewer, keys, model, data, addresses, names, path, source_path, speed, playlist_mode):
    times, positions, rotations, joints = load_trajectory(path, names)
    source = load_source(source_path, path, times, positions) if source_path else None
    period = times[-1] + times[-1] - times[-2]

    def set_frame(index):
        data.qpos[:3] = positions[index]
        data.qpos[3:7] = rotations[index]
        data.qpos[addresses] = joints[index]
        data.time = times[index]
        mujoco.mj_forward(model, data)

    with viewer.lock():
        set_frame(0)
        viewer.user_scn.ngeom = 0
        viewer.cam.lookat[:] = positions[0]
        viewer.cam.distance = 2.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15
        if source is not None:
            viewer.cam.lookat[:] = (positions[0] + source[0][0, 0]) / 2
            viewer.cam.distance = 4.5
            viewer.cam.azimuth = 0
            draw_source(viewer.user_scn, source[0][0], source[1], positions[0])
    viewer.sync()
    elapsed = 0.0
    previous = time.monotonic()
    paused = False
    while viewer.is_running():
        now = time.monotonic()
        if not paused:
            elapsed += (now - previous) * speed
            if elapsed >= period and not playlist_mode:
                elapsed %= period
        previous = now
        while not keys.empty():
            key = keys.get()
            if key in (ord('P'), ord('p')):
                paused = not paused
            elif playlist_mode and key in (ord('N'), ord('n')):
                return
            elif key in (ord('R'), ord('r')):
                elapsed = 0.0
        index = min(int(np.searchsorted(times, elapsed, side='right') - 1), len(times)-1)
        with viewer.lock():
            set_frame(index)
            if source is not None:
                viewer.user_scn.ngeom = 0
                draw_source(viewer.user_scn, source[0][index], source[1], positions[index])
        viewer.sync()
        if playlist_mode and elapsed >= period:
            return
        time.sleep(0.001)


if __name__ == '__main__':
    main()
