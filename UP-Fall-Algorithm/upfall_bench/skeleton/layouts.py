"""Joint names and bone lists for the skeleton formats in use (needed by ST-GCN's graph)."""
from __future__ import annotations

import numpy as np

COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]
COCO17_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

# RA-Pak-Edy privacy pipeline: COCO-17 without the 5 face points
BODY12 = COCO17[5:]
BODY12_EDGES = [(a - 5, b - 5) for a, b in COCO17_EDGES if a >= 5 and b >= 5]

MEDIAPIPE33 = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
               "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
               "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
               "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
               "right_heel", "left_foot_index", "right_foot_index"]
MEDIAPIPE33_EDGES = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12),
                     (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16),
                     (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                     (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

LAYOUTS = {"coco17": (COCO17, COCO17_EDGES), "body12": (BODY12, BODY12_EDGES),
           "mediapipe33": (MEDIAPIPE33, MEDIAPIPE33_EDGES)}


def guess_layout(n_joints: int) -> str | None:
    return {17: "coco17", 12: "body12", 33: "mediapipe33"}.get(n_joints)


def adjacency(n_joints: int, layout: str | None = None) -> np.ndarray:
    """Normalised adjacency with self loops, D^-1/2 (A + I) D^-1/2. Unknown layout -> chain."""
    layout = layout or guess_layout(n_joints)
    edges = LAYOUTS[layout][1] if layout in LAYOUTS else [(i, i + 1) for i in range(n_joints - 1)]
    a = np.eye(n_joints, dtype=np.float32)
    for i, j in edges:
        a[i, j] = a[j, i] = 1
    d = 1 / np.sqrt(a.sum(1))
    return a * d[:, None] * d[None, :]
