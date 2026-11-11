# Motion Generator

Native Unitree G1 23DoF reference trajectories from BONES-SEED BVH using SOMA Retargeter.

**BONES-SEED BVH → SOMA retargeting → G1 23DoF trajectory and check report.**

The flat-ground library covers standing, walking, turning and upper-body motion. The current training list contains 8,827 kinematically screened trajectories from 22,491 selected motions.

## Replay the library

From the repository root, with the local BVHs and trajectories installed:

```bash
conda activate motion-generator
python tools/replay.py --all
```

The viewer plays the human and robot side by side. Press **N** for the next motion, **P** to pause or resume, and **R** to restart the current motion.

- [Architecture and data format](docs/architecture.md)
- [Setup, generation, audit and single-motion replay](docs/usage.md)

Motion Data by [Bones Studio](https://bones.studio/). Data use is subject to the [BONES-SEED license](https://huggingface.co/datasets/bones-studio/seed/blob/main/LICENSE.md). Unitree and SOMA licenses remain with their assets. Large motion files stay outside Git.
