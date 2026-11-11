"""Compare SOMA configurations on the three authorized clips, preserving library results."""
import argparse
import copy
import json
from pathlib import Path
from check import ROOT, DEFAULT_ROBOT
from retarget import retarget_one, sha256

CLIPS = ('raise_your_hand_R_001__A533', 'warm_up_arms_002__A362',
         'crossed_arms_idle_turn_270_R_001__A456')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--arm-rotation', action='store_true', help='Pure SOMA: compare upper-arm rotation mapping and orientation-priority arm targets.')
    options.add_argument('--convergence', action='store_true', help='Pure SOMA: compare 24 and 96 IK iterations and observe final output clamping.')
    options.add_argument('--arm-postprocess', action='store_true', help='Compare corrected initialization with SOMA arm limb stabilization.')
    options.add_argument('--arm-segments-elbows', action='store_true', help='Arm-only adaptation with stronger elbow position targets.')
    options.add_argument('--arm-segments-limits', action='store_true', help='Arm-only adaptation with stronger model joint-limit objective.')
    options.add_argument('--arm-segments', action='store_true', help='Test arm-only bone length adaptation while preserving trunk and leg input.')
    options.add_argument('--arm-scale', action='store_true', help='Test the isolated arm-length calibration with high hand-position weight.')
    args = parser.parse_args()
    robot = DEFAULT_ROBOT
    base = json.loads((robot/'retarget/retargeter.json').read_text())
    variants = {}
    cfg = copy.deepcopy(base)
    for side in ('Left', 'Right'):
        cfg['ik_match_table'][side+'Hand']['t_weight'] = 100.0
    variants['hand_position_weight'] = cfg
    cfg = copy.deepcopy(base)
    for side in ('Left', 'Right'):
        for part in ('Arm', 'ForeArm', 'Hand'):
            cfg['ik_match_table'][side+part]['r_weight'] = 0.1
    variants['arm_orientation_weight'] = cfg
    if args.arm_scale:
        config = copy.deepcopy(variants['hand_position_weight'])
        config['human_robot_scaler_config'] = 'retarget/comparisons/arm_length_scaler.json'
        variants = {'arm_length_scale': config}
    if args.arm_segments or args.arm_segments_limits or args.arm_segments_elbows:
        config = json.loads((robot/'retarget/comparisons/arm_segments.json').read_text())
        if args.arm_segments_limits:
            config['joint_limit_weight'] = 100.0
        if args.arm_segments_elbows:
            for side in ('Left', 'Right'):
                config['ik_match_table'][side+'ForeArm']['t_weight'] = 100.0
        variants = {'arm_segments_elbows' if args.arm_segments_elbows else 'arm_segments_limits' if args.arm_segments_limits else 'arm_segments': config}
    if args.arm_postprocess:
        variants = {name: json.loads((robot/'retarget/comparisons'/f'{name}.json').read_text())
                    for name in ['arm_segments_init', 'arm_stabilizer']}
    if args.convergence:
        variants = {}
        for iterations in (24, 96):
            config = copy.deepcopy(base)
            config['ik_iterations'] = iterations
            config['record_clamp_diagnostics'] = True
            variants[f'pure_soma_iterations_{iterations}'] = config
    if args.arm_rotation:
        config = copy.deepcopy(base)
        config['record_clamp_diagnostics'] = True
        for side in ('Left', 'Right'):
            config['ik_match_table'][side+'Arm']['r_body'] = side.lower()+'_shoulder_yaw_link'
        variants = {'pure_soma_upper_arm_rotation': config}
        orientation = copy.deepcopy(config)
        for side in ('Left', 'Right'):
            for part in ('Arm', 'ForeArm', 'Hand'):
                orientation['ik_match_table'][side+part]['t_weight'] = 0.0
            orientation['ik_match_table'][side+'Hand']['r_weight'] = 0.1
        variants['pure_soma_arm_orientation'] = orientation
    protected = [ROOT/'motions/g1_23dof'/name for name in
                 ('library.jsonl', 'library_status.json', 'selected_library.jsonl')]
    protected += list((robot/'retarget').glob('*.json'))
    for stem in CLIPS:
        protected += [ROOT/'motions/g1_23dof/bones_seed'/f'{stem}.npz',
                      robot/'verification/bones_seed'/f'{stem}.json',
                      robot/'verification/representatives'/f'{stem}.json']
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    results = []
    for variant, config in variants.items():
        for stem in CLIPS:
            report_path = robot/'verification/comparisons'/variant/'bones_seed'/f'{stem}.json'
            output = ROOT/'motions/g1_23dof/comparisons'/variant/'bones_seed'/f'{stem}.npz'
            if not report_path.exists() and not output.exists():
                retarget_one(ROOT/'motions/source/bones_seed'/f'{stem}.bvh',
                             experiment=variant, config=config)
            report = json.loads(report_path.read_text())
            if report['provenance'].get('experimental_retarget_config') != config:
                raise ValueError('Existing comparison configuration differs; refusing replacement.')
            if report['passed'] != output.exists():
                raise ValueError('Incomplete comparison output requires inspection.')
            results.append({'variant': variant, 'source': stem,
                            'report': str(report_path.relative_to(ROOT)),
                            'report_sha256': sha256(report_path),
                            'output_sha256': sha256(output) if output.exists() else None,
                            'target_errors': report['target_errors'],
                            'metrics': report['metrics'], 'warnings': report['warnings']})
    for name, digest in hashes.items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Protected original changed: {name}')
    summary = {'scope': 'Three authorized sources only; isolated experimental configs. Model and production configs unchanged.',
               'protected_sha256': hashes, 'results': results, 'bulk_processing_allowed': False}
    if not args.arm_scale and not args.arm_segments and not args.arm_segments_limits and not args.arm_segments_elbows and not args.arm_postprocess and not args.convergence and not args.arm_rotation:
        summary['scope'] = 'Three authorized sources only; two independent weight changes. Targets, model and production configs unchanged.'
    path = robot/'verification/comparisons'/('pure_soma_arm_rotation.json' if args.arm_rotation else 'pure_soma_convergence.json' if args.convergence else 'arm_postprocess_comparison.json' if args.arm_postprocess else 'arm_segments_elbows_comparison.json' if args.arm_segments_elbows else 'arm_segments_limits_comparison.json' if args.arm_segments_limits else 'arm_segments_comparison.json' if args.arm_segments else 'arm_scale_comparison.json' if args.arm_scale else 'weight_comparison.json')
    text = json.dumps(summary, indent=2, allow_nan=False)+'\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Comparison summary already exists with different content.')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(text)
    print(f'Comparison saved: {path}')


if __name__ == '__main__':
    main()
