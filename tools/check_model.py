"""Load the native G1 23DoF model and check dimensions and forward kinematics."""
import argparse

import mujoco
import numpy as np

from check import DEFAULT_ROBOT, model_and_names


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    model, _, names = model_and_names(DEFAULT_ROBOT)
    dimensions = (model.nq, model.nv, model.nu, model.nmesh)
    if dimensions != (30, 29, 23, 29):
        raise ValueError(f'Unexpected model dimensions: {dimensions}')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.xpos).all() or not np.isfinite(data.xquat).all():
        raise ValueError('Non-finite body transforms.')
    print(f'Model OK: {len(names)} joints, {model.nmesh} meshes; nq={model.nq}, nv={model.nv}.')


if __name__ == '__main__':
    main()
