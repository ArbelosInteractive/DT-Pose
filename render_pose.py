JOINT_NAMES = [
    "pelvis",
    "spine",
    "neck",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_hip",
    "l_knee",
    "r_hip",
    "r_knee",
]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


BONES = [
    (0, 1), (1, 2), (2, 3),        # torso
    (2, 4), (4, 5), (5, 6),        # left arm
    (2, 7), (7, 8), (8, 9),        # right arm
    (0,10), (10,11),               # left leg
    (0,12), (12,13),               # right leg
]

def render_pose_3d(
    pose,
    root_idx=0,
    scale=50.0,
    swap_yz=True,
    flip_z=True,
    ax=None,
):
    """
    pose: (J, 3) numpy array
    """

    pose = np.asarray(pose).copy()

    # -----------------------------
    # 1. root-center
    # -----------------------------
    pose -= pose[root_idx]

    # -----------------------------
    # 2. scale to human size
    # -----------------------------
    pose *= scale

    # -----------------------------
    # 3. axis fix (WiFi datasets)
    # -----------------------------
    if swap_yz:
        pose = pose[:, [0, 2, 1]]

    if flip_z:
        pose[:, 2] *= -1

    # -----------------------------
    # 4. setup plot
    # -----------------------------
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # joints
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=40)

    # # bones
    # for i, j in BONES:
    #     ax.plot(
    #         [pose[i, 0], pose[j, 0]],
    #         [pose[i, 1], pose[j, 1]],
    #         [pose[i, 2], pose[j, 2]],
    #         linewidth=2,
    #     )

    # -----------------------------
    # 5. equal aspect ratio
    # -----------------------------
    center = pose.mean(axis=0)
    radius = np.linalg.norm(pose - center, axis=1).max()

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax
