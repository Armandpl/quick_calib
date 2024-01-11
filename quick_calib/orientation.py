"""Vectorized functions that transform between rotation matrices, euler angles and quaternions.

All support lists, array or array of arrays as inputs. Supports both x2y and y_from_x format
(y_from_x preferred!).
"""

import numpy as np
from numpy import array, dot, inner, linalg


def euler2quat(eulers):
    eulers = array(eulers)
    if len(eulers.shape) > 1:
        output_shape = (-1, 4)
    else:
        output_shape = (4,)
    eulers = np.atleast_2d(eulers)
    gamma, theta, psi = eulers[:, 0], eulers[:, 1], eulers[:, 2]

    q0 = np.cos(gamma / 2) * np.cos(theta / 2) * np.cos(psi / 2) + np.sin(gamma / 2) * np.sin(
        theta / 2
    ) * np.sin(psi / 2)
    q1 = np.sin(gamma / 2) * np.cos(theta / 2) * np.cos(psi / 2) - np.cos(gamma / 2) * np.sin(
        theta / 2
    ) * np.sin(psi / 2)
    q2 = np.cos(gamma / 2) * np.sin(theta / 2) * np.cos(psi / 2) + np.sin(gamma / 2) * np.cos(
        theta / 2
    ) * np.sin(psi / 2)
    q3 = np.cos(gamma / 2) * np.cos(theta / 2) * np.sin(psi / 2) - np.sin(gamma / 2) * np.sin(
        theta / 2
    ) * np.cos(psi / 2)

    quats = array([q0, q1, q2, q3]).T
    for i in range(len(quats)):
        if quats[i, 0] < 0:
            quats[i] = -quats[i]
    return quats.reshape(output_shape)


def quat2euler(quats):
    quats = array(quats)
    if len(quats.shape) > 1:
        output_shape = (-1, 3)
    else:
        output_shape = (3,)
    quats = np.atleast_2d(quats)
    q0, q1, q2, q3 = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    gamma = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    theta = np.arcsin(2 * (q0 * q2 - q3 * q1))
    psi = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    eulers = array([gamma, theta, psi]).T
    return eulers.reshape(output_shape)


def quat2rot(quats):
    quats = array(quats)
    input_shape = quats.shape
    quats = np.atleast_2d(quats)
    Rs = np.zeros((quats.shape[0], 3, 3))
    q0 = quats[:, 0]
    q1 = quats[:, 1]
    q2 = quats[:, 2]
    q3 = quats[:, 3]
    Rs[:, 0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    Rs[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    Rs[:, 0, 2] = 2 * (q0 * q2 + q1 * q3)
    Rs[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    Rs[:, 1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    Rs[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
    Rs[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    Rs[:, 2, 1] = 2 * (q0 * q1 + q2 * q3)
    Rs[:, 2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

    if len(input_shape) < 2:
        return Rs[0]
    else:
        return Rs


def rot2quat(rots):
    input_shape = rots.shape
    if len(input_shape) < 3:
        rots = array([rots])
    K3 = np.empty((len(rots), 4, 4))
    K3[:, 0, 0] = (rots[:, 0, 0] - rots[:, 1, 1] - rots[:, 2, 2]) / 3.0
    K3[:, 0, 1] = (rots[:, 1, 0] + rots[:, 0, 1]) / 3.0
    K3[:, 0, 2] = (rots[:, 2, 0] + rots[:, 0, 2]) / 3.0
    K3[:, 0, 3] = (rots[:, 1, 2] - rots[:, 2, 1]) / 3.0
    K3[:, 1, 0] = K3[:, 0, 1]
    K3[:, 1, 1] = (rots[:, 1, 1] - rots[:, 0, 0] - rots[:, 2, 2]) / 3.0
    K3[:, 1, 2] = (rots[:, 2, 1] + rots[:, 1, 2]) / 3.0
    K3[:, 1, 3] = (rots[:, 2, 0] - rots[:, 0, 2]) / 3.0
    K3[:, 2, 0] = K3[:, 0, 2]
    K3[:, 2, 1] = K3[:, 1, 2]
    K3[:, 2, 2] = (rots[:, 2, 2] - rots[:, 0, 0] - rots[:, 1, 1]) / 3.0
    K3[:, 2, 3] = (rots[:, 0, 1] - rots[:, 1, 0]) / 3.0
    K3[:, 3, 0] = K3[:, 0, 3]
    K3[:, 3, 1] = K3[:, 1, 3]
    K3[:, 3, 2] = K3[:, 2, 3]
    K3[:, 3, 3] = (rots[:, 0, 0] + rots[:, 1, 1] + rots[:, 2, 2]) / 3.0
    q = np.empty((len(rots), 4))
    for i in range(len(rots)):
        _, eigvecs = linalg.eigh(K3[i].T)
        eigvecs = eigvecs[:, 3:]
        q[i, 0] = eigvecs[-1]
        q[i, 1:] = -eigvecs[:-1].flatten()
        if q[i, 0] < 0:
            q[i] = -q[i]

    if len(input_shape) < 3:
        return q[0]
    else:
        return q


def euler2rot(eulers):
    return rotations_from_quats(euler2quat(eulers))


def rot2euler(rots):
    return quat2euler(quats_from_rotations(rots))


quats_from_rotations = rot2quat
quat_from_rot = rot2quat
rotations_from_quats = quat2rot
rot_from_quat = quat2rot
rot_from_quat = quat2rot
euler_from_rot = rot2euler
euler_from_quat = quat2euler
rot_from_euler = euler2rot
quat_from_euler = euler2quat


def quat_product(q, r):
    t = np.zeros(4)
    t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    return t


def rot_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rr = array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    rp = array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    ry = array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return ry.dot(rp.dot(rr))


def rot(axis, angle):
    # Rotates around an arbitrary axis
    ret_1 = (1 - np.cos(angle)) * array(
        [
            [axis[0] ** 2, axis[0] * axis[1], axis[0] * axis[2]],
            [axis[1] * axis[0], axis[1] ** 2, axis[1] * axis[2]],
            [axis[2] * axis[0], axis[2] * axis[1], axis[2] ** 2],
        ]
    )
    ret_2 = np.cos(angle) * np.eye(3)
    ret_3 = np.sin(angle) * array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return ret_1 + ret_2 + ret_3
