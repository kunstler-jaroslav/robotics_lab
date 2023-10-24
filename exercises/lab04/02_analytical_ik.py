#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-09-20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from copy import deepcopy
from tests.utils import sample_planar_manipulator
import numpy as np

import matplotlib
from robotics_toolbox.core import SO2
from robotics_toolbox.core import SE2
from robotics_toolbox.render import RendererPlanar
from robotics_toolbox.render.planar_manipulator_renderer import (
    PlanarManipulatorRenderer,
)
from robotics_toolbox.robots import PlanarManipulator
from robotics_toolbox.utils import save_fig

# robot = PlanarManipulator(
#         link_lengths=np.random.uniform(0.4, 0.8, size=3),
#         base_pose=SE2(
#             translation=np.random.uniform(-0.5, 0.5, size=2),
#             rotation=SO2(np.random.uniform(-np.pi, np.pi)),
#         ))
# robot.structure = "RRR"

robot = PlanarManipulator(
    link_lengths=[0.5, 0.5, 0.5],
    base_pose=SE2([-0.9, -0.75]),
    structure="RRR",
    # structure="PRR", # for optional HW
)
robot.q = np.random.uniform(-np.pi, np.pi, size=robot.dof)
render = RendererPlanar(lim_scale=2.0)

desired_pose = robot.flange_pose()
# desired_pose = SE2(translation=[-0.75, 0.25], rotation=SO2(angle=np.deg2rad(45 + 90)))

render.plot_se2(desired_pose)

solutions = robot.ik_analytical(desired_pose)

# # todo
# render = RendererPlanar(lim_scale=2.0)
# robot = sample_planar_manipulator(3)
# robot.structure = "RRR"
# robot.q = np.random.uniform(-np.pi, np.pi, size=robot.dof)
# target = robot.flange_pose()
# render.plot_se2(target)
# solutions = robot.ik_analytical(target)
# # todo


if len(solutions) == 0:
    print("No solution found.")

# Display all solutions with different colors
clrs = matplotlib.colormaps["tab10"].colors
for i, sol in enumerate(solutions):
    r = deepcopy(robot)
    r.q = sol
    render.manipulators[r] = PlanarManipulatorRenderer(render.ax, r, color=clrs[i])
    render.plot_manipulator(r)
    save_fig()

render.wait_for_close()
