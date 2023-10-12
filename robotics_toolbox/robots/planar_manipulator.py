#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-08-21
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from shapely import MultiPolygon, LineString, MultiLineString

from robotics_toolbox.core import SE2, SE3, SO2
from robotics_toolbox.robots.robot_base import RobotBase


class PlanarManipulator(RobotBase):
    def __init__(
            self,
            link_lengths: ArrayLike | None = None,
            structure: list[str] | str | None = None,
            base_pose: SE2 | None = None,
            gripper_length: float = 0.2,
    ) -> None:
        """
        Creates a planar manipulator composed by rotational and prismatic joints.
        The manipulator kinematics is defined by following kinematics chain:
         T_flange = (T_base) T(q_0) T(q_1) ... T_n(q_n),
        where
         T_i describes the pose of the next link w.r.t. the previous link computed as:
         T_i = R(q_i) Tx(l_i) if joint is revolute,
         T_i = R(l_i) Tx(q_i) if joint is prismatic,
        with
         l_i is taken from @param link_lengths;
         type of joint is defined by the @param structure.

        Args:
            link_lengths: either the lengths of links attached to revolute joints in [m]
                or initial rotation of prismatic joint [rad].
            structure: sequence of joint types, either R or P, [R]*n by default
            base_pose: mounting of the robot, identity by default
            gripper_length: length of the gripper measured from the flange
        """
        super().__init__()
        self.link_lengths: np.ndarray = np.asarray(
            [0.5] * 3 if link_lengths is None else link_lengths
        )
        n = len(self.link_lengths)
        self.base_pose = SE2() if base_pose is None else base_pose
        self.structure = ["R"] * n if structure is None else structure
        assert len(self.structure) == len(self.link_lengths)
        self.gripper_length = gripper_length

        # Robot configuration:
        self.q = np.array([np.pi / 8] * n)
        self.gripper_opening = 0.2

        # Configuration space
        self.q_min = np.array([-np.pi] * n)
        self.q_max = np.array([np.pi] * n)

        # Additional obstacles for collision checking function
        self.obstacles: MultiPolygon | None = None

    @property
    def dof(self):
        """Return number of degrees of freedom."""
        return len(self.q)

    def sample_configuration(self):
        """Sample robot configuration inside the configuration space. Will change
        internal state."""
        return np.random.uniform(self.q_min, self.q_max)

    def set_configuration(self, configuration: np.ndarray | SE2 | SE3):
        """Set configuration of the robot, return self for chaining."""
        self.q = configuration
        return self

    def configuration(self) -> np.ndarray | SE2 | SE3:
        """Get the robot configuration."""
        return self.q

    def flange_pose(self) -> SE2:
        """Return the pose of the flange(where connected to the link) in the reference frame."""
        # HW02: implement fk for the flange
        flange_pose = self.base_pose
        for i in range(len(self.q)):
            if self.structure[i] == "R":
                rotation_matrix = SE2(rotation=SO2(float(self.q[i])))
                flange_pose = flange_pose * rotation_matrix * SE2(translation=[self.link_lengths[i], 0])
            elif self.structure[i] == "P":
                translation_matrix = SE2(rotation=SO2(float(self.link_lengths[i])))
                flange_pose = flange_pose * translation_matrix * SE2(translation=[self.q[i], 0])
        return flange_pose

    def fk_all_links(self) -> list[SE2]:
        """Compute FK for frames that are attached to the links of the robot.
        The first frame is base_frame, the next frames are described in the constructor.
        returns list of SE2 transformations
        """
        # HW02: implement fk
        current_frame = self.base_pose
        frames = [current_frame]

        for i in range(len(self.q)):
            if self.structure[i] == "R":  # Revolute joint
                rotation_matrix = SE2(rotation=SO2(float(self.q[i])))
                current_frame = current_frame * rotation_matrix * SE2(translation=[self.link_lengths[i], 0])
            elif self.structure[i] == "P":  # Prismatic joint
                translation_matrix = SE2(rotation=SO2(float(self.link_lengths[i])))
                current_frame = current_frame * translation_matrix * SE2(translation=[self.q[i], 0])

            frames.append(current_frame)

        return frames

    def _gripper_lines(self, flange: SE2):
        """Return tuple of lines (start-end point) that are used to plot gripper
        attached to the flange frame."""
        gripper_opening = self.gripper_opening / 2.0
        return (
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([0, +gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([self.gripper_length, -gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, +gripper_opening])).translation,
                (flange * SE2([self.gripper_length, +gripper_opening])).translation,
            ),
        )

    def get_to_flange_transformations(self):
        """
        Computes transformation from each joint to the flange
        """
        frames = self.fk_all_links()
        base_flange = self.flange_pose()
        to_flange = []
        for frame in frames:
            to_flange.append(frame.inverse() * base_flange)
        return to_flange

    def jacobian(self) -> np.ndarray:
        """Computes jacobian of the manipulator for the given structure and
        configuration."""
        jac = np.zeros((3, len(self.q)))
        angle = np.pi / 2
        rotation_90 = np.array([[0, -1], [1, 0]])
        # HW03 implement jacobian computation
        # by columns
        # J = N x M  --  N = 3, M = len(self.q))
        frames = self.fk_all_links()
        to_flange = self.get_to_flange_transformations()
        for i in range(len(self.q)):
            if self.structure[i] == "R":
                # Tlf = T^(-1) * Tf - transformation from Tl do T flange (Tl - base - Tf) how does tl transfers to tf
                # n = (90def) * Tlf
                # jac[0:2, i] = Rw,j * n
                n = np.dot(rotation_90, to_flange[i].translation)
                add = np.dot(frames[i].rotation.rot, n)
                jac[0:2, i] = add
                jac[2, i] = 1  # 1 for rotation joints
            if self.structure[i] == "P":
                #  jac[0:2, i] = Rw,j+1 * a , a = translation axis => a = [1, 0]
                jac[0:2, i] = np.dot(frames[i+1].rotation.rot, [1, 0])
                # jac[0:2, i] = np.dot(to_flange[i].rotation.rot, [1, 0])
                jac[2, i] = 0  # 0 for translation joints

        return jac

    def jacobian_finite_difference(self, delta=1e-5) -> np.ndarray:
        """
        Using finite differentiation compute jacobian from rotation matrix and translation
        matrix of SE2 transformation, were R = self.flange_pose().rotation.rot
        and T = self.flange_pose().translation
        """

        jac = np.zeros((3, len(self.q)))

        # Compute the unperturbed end-effector pose
        flange_pose = self.flange_pose()
        original_translation = flange_pose.translation
        original_rotation = flange_pose.rotation.rot
        q_backup = self.q.copy()

        x_original = original_translation[0]
        y_original = original_translation[1]
        theta_original = np.arctan2(original_rotation[1, 0], original_rotation[0, 0])

        for i in range(len(self.q)):
            # Perturb the joint angle
            q_perturbed = self.q.copy()
            q_perturbed[i] += delta
            # Set the perturbed joint configuration
            self.set_configuration(q_perturbed)

            # Compute the perturbed end-effector pose
            perturbed_flange_pose = self.flange_pose()
            perturbed_translation = perturbed_flange_pose.translation
            perturbed_rotation = perturbed_flange_pose.rotation.rot

            x_perturbed = perturbed_translation[0]
            y_perturbed = perturbed_translation[1]
            theta_perturbed = np.arctan2(perturbed_rotation[1, 0], perturbed_rotation[0, 0])

            x_data = (x_perturbed - x_original) / delta
            y_data = (y_perturbed - y_original) / delta
            theta_data = (theta_perturbed - theta_original) / delta

            jac[:, i] = [x_data, y_data, theta_data]
            # Restore the original joint configuration
            self.set_configuration(q_backup)
        return jac

    def ik_numerical(
            self,
            flange_pose_desired: SE2,
            max_iterations=1000,
            acceptable_err=1e-4,
    ) -> bool:
        """Compute IK numerically. Value self.q is used as an initial guess and updated
        to solution of IK. Returns True if converged, False otherwise."""
        # todo: HW04 implement numerical IK

        return False

    def ik_analytical(self, flange_pose_desired: SE2) -> list[np.ndarray]:
        """Compute IK analytically, return all solutions for joint limits being
        from -pi to pi for revolute joints -inf to inf for prismatic joints."""
        assert self.structure in (
            "RRR",
            "PRR",
        ), "Only RRR or PRR structure is supported"

        # todo: HW04 implement analytical IK for RRR manipulator
        # todo: HW04 optional implement analytical IK for PRR manipulator
        if self.structure == "RRR":
            pass
        return []

    def in_collision(self) -> bool:
        """Check if robot in its current pose is in collision."""
        frames = self.fk_all_links()
        points = [f.translation for f in frames]
        gripper_lines = self._gripper_lines(frames[-1])

        links = [LineString([a, b]) for a, b in zip(points[:-2], points[1:-1])]
        links += [MultiLineString((*gripper_lines, (points[-2], points[-1])))]
        for i in range(len(links)):
            for j in range(i + 2, len(links)):
                if links[i].intersects(links[j]):
                    return True
        return MultiLineString(
            (*gripper_lines, *zip(points[:-1], points[1:]))
        ).intersects(self.obstacles)
