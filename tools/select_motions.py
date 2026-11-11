"""Classify every BONES-SEED metadata row; never generate or delete motion."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['filename', 'content_name', 'category', 'content_type_of_movement',
          'content_body_position', 'content_props', 'content_uniform_style',
          'content_technical_description', 'content_natural_desc_1', 'is_mirror', 'actor_uid']
EXCLUDED = r'\b(jog\w*|run|running|sprint\w*|jump\w*|hop|hopping|danc\w*|flip\w*|somersault\w*|cartwheel\w*|handstand\w*|roll|rolling|crawl\w*|stairs?|ladder|climb\w*)\b'


def words(value):
    return value.lower().replace('_', ' ').replace('-', ' ')



def upper_scope(name, movement, desc):
    """Require action identity and description evidence for the expanded upper-body scope."""
    action = r'\b(pick|picking|grab|grabbing|grasp|grasping|take|taking|put|putting|hold|holding|carry|carrying|push|pushing|pull|pulling|itching|scratch|scratching)\b'
    named = bool(re.search(action, name) or 'crossed arms' in name)
    described = bool(re.search(action, desc) or re.search(r'arms? (crossed|folded)|cross(ing)? (their |the |both )?arms', desc))
    manipulation = ('object manipulation' in movement and
                    re.search(r'\b(hands?|object|item)\b', name) and
                    re.search(r'\b(hands?|object|item)\b', desc))
    movement_parts = {part.strip() for part in movement.split(',')}
    compatible = movement_parts <= {'standing', 'standing idle', 'gesture', 'action',
        'transition', 'object manipulation', 'picking up item', 'pulling', 'pushing',
        'pressing', 'interacting', 'switching'}
    return bool(compatible and ((named and described) or manipulation))


def review_flags(row):
    """Scope retention does not establish native-23DoF feasibility or object interaction."""
    flags = ['native_23dof_motion_fit_unverified']
    text = words(row['content_name'] + ' ' + row['content_technical_description'] + ' ' + row['content_natural_desc_1'])
    if row['content_props'].strip().lower() not in {'0', '', 'none'}:
        flags.append('object_interaction_pose_only')
    if re.search(r'\b(bend\w*|lean\w*|stoop\w*|torso|waist|floor|ground)\b', text):
        flags.append('trunk_compensation_requires_review')
    return flags


def classify(row):
    """Ordered evidence rules. Unresolved rows remain review, never silently excluded."""
    name = words(row['content_name'].removesuffix('_M'))
    filename = words(row['filename'].split('__')[0])
    movement = words(row['content_type_of_movement']).strip()
    body = words(row['content_body_position']).strip()
    desc = words(row['content_technical_description'] + ' ' + row['content_natural_desc_1'])
    props = row['content_props'].strip().lower()
    style = row['content_uniform_style'].strip().lower()
    banned_name = bool(re.search(EXCLUDED, name))
    banned_move = bool(re.search(EXCLUDED, movement))
    banned_file = bool(re.search(EXCLUDED, filename))
    if banned_file != banned_name:
        return 'review', [], 'filename_action_conflict'
    # Action identity conflicts are not resolved by choosing one metadata field.
    for vocabulary in [('start', 'stop', 'loop'), ('walk', 'jog', 'jump', 'reaching')]:
        named = {t for t in vocabulary if re.search(r'\b' + t + r'\b', name)}
        filed = {t for t in vocabulary if re.search(r'\b' + t + r'\b', filename)}
        if named and filed and named != filed:
            return 'review', [], 'filename_action_conflict'
    if banned_name and (banned_move or re.search(EXCLUDED, desc)):
        return 'exclude', [], 'excluded_action_confirmed'
    if banned_move or banned_name:
        return 'review', [], 'action_fields_disagree'
    if re.search(r'\b(injured|crutch|crutches)\b', name) and ('injured' in style or 'crutch' in props):
        return 'exclude', [], 'injured_or_assisted_locomotion'
    if any(x in body for x in ['sitting', 'kneeling', 'lying', 'crawling', 'all fours', 'plank', 'squating', 'crouching', 'croaching']):
        return 'review', [], 'nonstanding_or_mixed_posture'
    expanded_upper = upper_scope(name, movement, desc)
    if expanded_upper:
        if re.search(r'\b(aggress\w*|aggres\w*|violently|forcefully|heavy|heavily)\b', name + ' ' + desc):
            return 'review', [], 'load_or_force_requires_review'
        if style != 'neutral':
            return 'review', [], 'style_requires_review'
        if body in {'standing, leaning', 'standing, bent down'}:
            if not re.search(r'\b(slight\w*|mild\w*|lightly)\b', desc):
                return 'review', [], 'bend_extent_and_native_waist_fit_unverified'
        elif body != 'standing':
            return 'review', [], 'posture_requires_review'
        if row['content_complex_action'].strip() not in {'0', '0.0', ''}:
            return 'review', [], 'composite_action'
        return 'keep', ['upper_body'], 'approved_upper_scope_with_supporting_description'
    if props not in {'0', '', 'none'}:
        return 'review', [], 'object_or_environment_dependency'
    if style != 'neutral':
        return 'review', [], 'style_requires_review'
    if body != 'standing':
        return 'review', [], 'posture_requires_review'
    if row['content_complex_action'].strip() not in {'0', '0.0', ''}:
        return 'review', [], 'composite_action'
    if re.search(r'\b(dog|car|crowd|obstacle|painful|talking|quip|pull|push|hit|eating|book|mosquito|itching|rubbing|dusting|kissing|blow|stretch)\b', name):
        return 'review', [], 'interaction_or_stylized_action'
    labels = []
    if re.fullmatch(r'(neutral )?(walk( ff)?( (loop|start|stop))? [0-9]{3}( [rl])?( (normal pace|slow|very slow))?|walk arc (cw|ccw) (loop|start|stop)( [rl])?|turn walk [0-9]{3}( [rl])?|walk [0-9]{3} turn [0-9]{3} long( [rl])?)', name) and 'walking' in movement and re.search(r'\bwalk\w*\b', desc):
        if re.search(r'\b(fast|hurry|exaggerated|amplified|weak|sick)\b', name + ' ' + desc):
            return 'review', [], 'walking_speed_or_style'
        labels.append('walking')
    if re.search(r'\bturn\w*\b', name) and ('turning' in movement or 'walking' in movement) and re.search(r'\bturn\w*\b', desc):
        labels.append('turning')
    if re.fullmatch(r'(neutral )?idle( loop)?', name) and movement in {'standing idle', 'standing', 'exercising'} and re.search(r'\bstand\w*\b', desc):
        labels.append('standing')
    upper_name = re.match(r'^(reaching |raise your hand |wave |point |warm up arms$|calm down |the ring |wait )', name)
    upper_desc = re.search(r'\b(arms?|hands?|reach\w*|point\w*)\b', desc)
    if upper_name and upper_desc and movement in {'gesture', 'pointing', 'standing', 'action', 'exercising', 'waving', 'standing, pointing'}:
        if re.search(r'\b(throw\w*|punch\w*|hit\w*|clap\w*|fingers?|snap\w*|fight\w*)\b', desc):
            return 'review', [], 'hand_detail_or_forceful_action'
        labels.append('upper_body')
    if labels:
        return 'keep', labels, 'name_description_and_attributes_agree'
    return 'review', [], 'unmapped_action_semantics'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=ROOT/'motions/source/bones_seed/metadata/seed_metadata_v004.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'motions/source/bones_seed/screening/expanded_upper_body')
    args = parser.parse_args()
    with args.metadata.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len({r['filename'] for r in rows}) == len(rows), 'Duplicate filenames'
    counts, labels, reasons, original = Counter(), Counter(), Counter(), Counter()
    examples = defaultdict(list)
    decisions = []
    coarse = Counter()
    for row in rows:
        state, tags, reason = classify(row)
        counts[state] += 1
        if row['is_mirror'].lower() != 'true':
            original[state] += 1
        if row['category'] in ['Basic Locomotion Neutral', 'Baseline', 'Gestures', 'Looking and Pointing']:
            coarse[row['category']] += 1
        if state == 'keep':
            labels.update(tags)
        if state == 'review':
            reasons[reason] += 1
        entry = {k: row[k] for k in FIELDS}
        entry.update(decision=state, labels=tags, reason=reason,
                     source='motions/source/bones_seed/' + row['filename'] + '.bvh',
                     review_flags=review_flags(row) if state == 'keep' else [])
        decisions.append(entry)
        for key in (tags if state == 'keep' else [reason]):
            if row['is_mirror'].lower() != 'true' and len(examples[key]) < 4:
                examples[key].append(entry)
    assert sum(counts.values()) == len(rows)
    ledger = ROOT/'motions/g1_23dof/library.jsonl'
    done = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                item = json.loads(line)
                done[item['source']] = item
            except json.JSONDecodeError:
                continue
    reuse = Counter(done.get(x['source'], {}).get('status', 'unrecorded') for x in decisions if x['decision'] == 'keep')
    summary = {'total': len(rows), 'counts': dict(counts), 'non_mirrored_counts': dict(original),
               'keep_label_counts_overlapping': dict(labels), 'keep_unique': counts['keep'],
               'review_reasons': dict(reasons), 'coarse_categories': dict(coarse),
               'keep_existing_ledger_status': dict(reuse),
               'keep_review_flags': dict(Counter(flag for x in decisions if x['decision'] == 'keep' for flag in x['review_flags'])),
               'metadata_sha256': hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
               'rules_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'scope': 'Metadata scope screening, not motion quality approval. No generation authorized by this output.',
               'examples': dict(examples)}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('decisions.jsonl', ''.join(json.dumps(x, ensure_ascii=True) + '\n' for x in decisions)),
                        ('summary.json', json.dumps(summary, indent=2, ensure_ascii=True) + '\n')]:
        target = args.output/name
        if target.exists() and target.read_text() != value:
            raise FileExistsError(f'Refusing to replace different screening results: {target}')
        target.write_text(value)
    print(json.dumps({k: v for k, v in summary.items() if k != 'examples'}, indent=2))


if __name__ == '__main__':
    main()
