# Operations

## 1. Activate

Use this for the existing workstation:

```bash
source /home/xiaopeng/miniforge3/etc/profile.d/conda.sh
conda activate motion-generator
cd /home/xiaopeng/workspace/motion-generator
```

## 2. Install on a New Machine

Skip this section if the environment already exists. Use Ubuntu with an NVIDIA GPU for retargeting; replay requires a desktop display.

```bash
git clone https://github.com/rockwxp/motion-generator.git
cd motion-generator
conda create -n motion-generator -c conda-forge python=3.12 pip -y
conda activate motion-generator
python -m pip install \
  "mujoco==3.4.0" "newton==1.3.0" "warp-lang==1.19.0" \
  "numpy==2.4.3" "scipy==1.17.1" "typing_extensions==4.15.0" \
  "huggingface_hub==2.4.0" "anyio==4.11.0" \
  "soma_retargeter @ git+https://github.com/NVIDIA/soma-retargeter.git@cbdb2b494a2af82137bf9b6f15122d60167281c4"
python -m pip check
```

The checkout must contain `tools/` and the complete `robots/g1_23dof/` assets. Check the model:

```bash
python tools/check_model.py
```

## Verify the Official 29DoF Pipeline

```bash
python tools/verify_official.py
```

This verifies three official BVH/CSV pairs (walking, waving, and picking up) from the installed SOMA revision. The bundled 29DoF model and settings remain unchanged. Files are checked against Git LFS hashes; completed results are reused. Official CSV units are decoded by SOMA itself. Human inputs: `motions/source/soma_official/`. Official and reproduced 29DoF trajectories: `motions/g1_29dof/official_reference/` and `official_validation/`. Read `official_validation/summary.json` for measured differences. This is a software/output check, not a physical execution test.

## 3. Select and Review

```bash
python tools/select_motions.py
python tools/retarget_dataset.py --status
```

Screening results: `motions/source/bones_seed/screening/expanded_upper_body/decisions.jsonl` and `summary.json`. Counts include actor and mirror variants. Metadata retention is not quality approval. Sources are already available on the existing workstation. On a new machine, prepare the sources (section 7) first. Restore the local `motions/source/bones_seed/representatives.jsonl` selection before running sections 4 and 6.

## 4. Representative Processing

```bash
python tools/retarget_dataset.py --selection motions/source/bones_seed/representatives.jsonl
```

The explicit representative list covers locomotion and upper-body actions. Existing trajectories and rejection reports are reused, never overwritten or retargeted. New sources use the same SOMA solver and per-motion checks as the selected library. The selected ledger is separate from the stopped full-library records.

## 5. Target Errors and Synchronized Replay

```bash
motion=walk_ff_loop_180_R_slow_001__A443
python tools/check.py "motions/g1_23dof/bones_seed/$motion.npz" \
  --source "motions/source/bones_seed/$motion.bvh"
python tools/replay.py "motions/g1_23dof/bones_seed/$motion.npz" \
  --source "motions/source/bones_seed/$motion.bvh" --speed 0.5
```

Play the full training list in one window, with each source BVH beside its robot trajectory:

```bash
python tools/replay.py --all
```

The viewer reads repository-relative NPZ paths from `robots/g1_23dof/verification/legacy_recheck/training_paths.txt`. Place the matching BVHs in `motions/source/bones_seed/` and NPZs in `motions/g1_23dof/bones_seed/`. Playback advances automatically and ends after the last clip. Press N for the next clip, P to pause or resume, and R to restart the current clip. Use `--speed 2` for double speed.

Optional `--report` on check.py writes a supplementary report to a new path; existing reports are never replaced. Source hashes, timing and robot configuration must match. Source-target extraction does not rerun IK. Playback uses FK, not physics stepping.

Historical comparison outputs and reports remain preserved under `motions/g1_23dof/comparisons/` and `robots/g1_23dof/verification/comparisons/`. They are not production settings. New generation uses pure SOMA with the native configuration; no custom bone adaptation or temporal filter is enabled.

## 6. Selected Library

Official 29DoF reproduction and native configuration checks are recorded in `robots/g1_23dof/verification/official_pipeline_review.json`. Processing requires those records and unchanged configuration hashes. Historical failed-motion reviews remain preserved; an individual failure does not block other selected files.

Reuse existing outcomes without IK, then process remaining selected files:

```bash
python tools/retarget_dataset.py --selection motions/source/bones_seed/screening/expanded_upper_body/decisions.jsonl --reuse-only
python tools/retarget_dataset.py --selection motions/source/bones_seed/screening/expanded_upper_body/decisions.jsonl --batch-size 4
python tools/retarget_dataset.py --selection motions/source/bones_seed/screening/expanded_upper_body/decisions.jsonl --audit
```

On the RTX 5090, eight reaching and walking BVHs (9,946 frames per group, paired with mirror variants) took 87.25 s with batch size 1 and 28.17 s with batch size 4, including checks and saving.

Only `keep` rows are processed. `--batch-size N` sends up to N compatible pending BVHs to one SOMA solve (default 1). Each result is checked and recorded separately. `--limit` bounds pending entries for validation. No selection means no generation. Existing rejections are not retried. Invalid source data is recorded as `skipped_error`; runtime/system failures remain visible. Sources with comparison-only results are recorded as `skipped_review`; processing continues without rerunning them. Inspect errors rather than deleting results to force reruns.

| Local artifact | Path |
| --- | --- |
| Robot trajectories | `motions/g1_23dof/bones_seed/` |
| Original check reports | `robots/g1_23dof/verification/bones_seed/` |
| Per-source outcomes | `motions/g1_23dof/selected_library.jsonl` |
| Supplementary checks for 1,215 legacy trajectories | `robots/g1_23dof/verification/legacy_recheck/` |

Use `training_paths.txt` for the screened library. The ledger preserves historical classifications, including three `reference` entries with warnings. The list includes 7,612 original unflagged trajectories and 1,215 that passed supplementary checks. The checks cover kinematics. Physical tracking and balance require simulation evaluation.

## 7. Source Preparation on a New Machine

```bash
hf auth login
hf auth whoami
python tools/download_dataset.py
```

Use an account authorized for BONES-SEED. This verifies/extracts the pinned source archive; it does not start retargeting. Model checking and all runtime commands use the dedicated motion-generator environment.
