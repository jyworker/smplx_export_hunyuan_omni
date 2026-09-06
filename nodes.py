import os
import json

import torch
import numpy as np
import smplx as smplx_lib


# ---------------------------------------------------------------------------
# SMPL-X joint indices (in the order returned by `SMPLXOutput.joints`).
# Indices 0-54 are the regressed body/hand/face joints; 55+ are the extra
# landmarks appended by smplx's VertexJointSelector (nose/eyes/ears, toes,
# heels and the five finger tips per hand).
# ---------------------------------------------------------------------------
J = {
    "pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3,
    "left_knee": 4, "right_knee": 5, "spine2": 6,
    "left_ankle": 7, "right_ankle": 8, "spine3": 9,
    "left_foot": 10, "right_foot": 11, "neck": 12,
    "left_collar": 13, "right_collar": 14, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
    "jaw": 22, "left_eye": 23, "right_eye": 24,
    # left hand
    "left_index1": 25, "left_index2": 26, "left_index3": 27,
    "left_middle1": 28, "left_middle2": 29, "left_middle3": 30,
    "left_pinky1": 31, "left_pinky2": 32, "left_pinky3": 33,
    "left_ring1": 34, "left_ring2": 35, "left_ring3": 36,
    "left_thumb1": 37, "left_thumb2": 38, "left_thumb3": 39,
    # right hand
    "right_index1": 40, "right_index2": 41, "right_index3": 42,
    "right_middle1": 43, "right_middle2": 44, "right_middle3": 45,
    "right_pinky1": 46, "right_pinky2": 47, "right_pinky3": 48,
    "right_ring1": 49, "right_ring2": 50, "right_ring3": 51,
    "right_thumb1": 52, "right_thumb2": 53, "right_thumb3": 54,
    # extra landmarks / tips
    "nose": 55,
    "left_big_toe": 60, "right_big_toe": 63,
    "left_thumb_tip": 66, "left_index_tip": 67, "left_middle_tip": 68,
    "left_ring_tip": 69, "left_pinky_tip": 70,
    "right_thumb_tip": 71, "right_index_tip": 72, "right_middle_tip": 73,
    "right_ring_tip": 74, "right_pinky_tip": 75,
}

# ---------------------------------------------------------------------------
# The 51 Hunyuan3D-Omni bones, in the exact order the reference
# demos/pose/*_bone.txt files use. Each entry maps a bone to (start, end)
# SMPL-X joints. Naming quirks are kept to match Hunyuan (e.g. "Foot.R" is
# actually the thigh, "Knee.R" the shin, "Ankle.R" the foot).
# ---------------------------------------------------------------------------
HUNYUAN_BONES = [
    ("Upper Body",   "pelvis",         "spine2"),
    ("Upper Body 2", "spine2",         "neck"),
    ("Neck",         "neck",           "head"),
    ("Head",         "head",           "nose"),
    ("Eye.R",        "right_eye",      "right_eye"),
    ("Eye.L",        "left_eye",       "left_eye"),

    ("Shoulder.R",   "right_collar",   "right_shoulder"),
    ("Arm.R",        "right_shoulder", "right_elbow"),
    ("Elbow.R",      "right_elbow",    "right_wrist"),
    ("Wrist.R",      "right_wrist",    "right_middle1"),
    ("Thumb0.R",     "right_wrist",    "right_thumb1"),
    ("Thumb1.R",     "right_thumb1",   "right_thumb2"),
    ("Thumb2.R",     "right_thumb2",   "right_thumb3"),
    ("Index1.R",     "right_index1",   "right_index2"),
    ("Index2.R",     "right_index2",   "right_index3"),
    ("Index3.R",     "right_index3",   "right_index_tip"),
    ("Middle1.R",    "right_middle1",  "right_middle2"),
    ("Middle2.R",    "right_middle2",  "right_middle3"),
    ("Middle3.R",    "right_middle3",  "right_middle_tip"),
    ("Ring1.R",      "right_ring1",    "right_ring2"),
    ("Ring2.R",      "right_ring2",    "right_ring3"),
    ("Ring3.R",      "right_ring3",    "right_ring_tip"),
    ("Pinky1.R",     "right_pinky1",   "right_pinky2"),
    ("Pinky2.R",     "right_pinky2",   "right_pinky3"),
    ("Pinky3.R",     "right_pinky3",   "right_pinky_tip"),

    ("Shoulder.L",   "left_collar",    "left_shoulder"),
    ("Arm.L",        "left_shoulder",  "left_elbow"),
    ("Elbow.L",      "left_elbow",     "left_wrist"),
    ("Wrist.L",      "left_wrist",     "left_middle1"),
    ("Thumb0.L",     "left_wrist",     "left_thumb1"),
    ("Thumb1.L",     "left_thumb1",    "left_thumb2"),
    ("Thumb2.L",     "left_thumb2",    "left_thumb3"),
    ("Index1.L",     "left_index1",    "left_index2"),
    ("Index2.L",     "left_index2",    "left_index3"),
    ("Index3.L",     "left_index3",    "left_index_tip"),
    ("Middle1.L",    "left_middle1",   "left_middle2"),
    ("Middle2.L",    "left_middle2",   "left_middle3"),
    ("Middle3.L",    "left_middle3",   "left_middle_tip"),
    ("Ring1.L",      "left_ring1",     "left_ring2"),
    ("Ring2.L",      "left_ring2",     "left_ring3"),
    ("Ring3.L",      "left_ring3",     "left_ring_tip"),
    ("Pinky1.L",     "left_pinky1",    "left_pinky2"),
    ("Pinky2.L",     "left_pinky2",    "left_pinky3"),
    ("Pinky3.L",     "left_pinky3",    "left_pinky_tip"),

    ("Lower Body",   "spine1",         "pelvis"),
    ("Foot.R",       "right_hip",      "right_knee"),
    ("Knee.R",       "right_knee",     "right_ankle"),
    ("Ankle.R",      "right_ankle",    "right_foot"),
    ("Foot.L",       "left_hip",       "left_knee"),
    ("Knee.L",       "left_knee",      "left_ankle"),
    ("Ankle.L",      "left_ankle",     "left_foot"),
]


def _as_tensor(value, shape, dtype=torch.float32):
    """Coerce a numpy array / tensor / list into a float tensor of `shape`.

    Handles CUDA tensors and tensors that require grad (both produced by the
    estimator) by moving to CPU / detaching before reshaping.
    """
    if value is None:
        return torch.zeros(shape, dtype=dtype)
    if torch.is_tensor(value):
        t = value.detach().to("cpu", dtype)
    else:
        t = torch.as_tensor(np.asarray(value)).to(dtype)
    return t.reshape(shape)


class SMPLxExportHunyuanOmni:
    """Convert a SMPL-X parameter dict (e.g. from ComfyUI-SMPLx-Estimator's
    MultiHMR node) into the 51-bone pose file consumed by Hunyuan3D-Omni's
    `--control_type pose`.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smplx": ("SMPLX",),
                "model_path": ("STRING", {"default": "models/smplx"}),
                "gender": (["neutral", "male", "female"], {"default": "neutral"}),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 0.9999, "min": 0.1, "max": 2.0, "step": 0.001}),
                "flip_y": ("BOOLEAN", {"default": False}),
                "flip_z": ("BOOLEAN", {"default": False}),
                "save_name": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("pose_bone_txt", "skeleton_bone_json", "saved_path")
    FUNCTION = "export"
    CATEGORY = "SMPLx"

    def export(self, smplx, model_path, gender,
               scale=0.9999, flip_y=False, flip_z=False, save_name=""):
        # `smplx` here is the incoming data dict; the smplx *library* is
        # imported at top level as `smplx_lib`, so there is no name clash.
        smplx_data = smplx if isinstance(smplx, dict) else dict(smplx)

        gender = smplx_data.get("gender", gender)
        model_path = smplx_data.get("model_path", model_path)

        # Build the SMPL-X body model. use_pca=False so full 45-D axis-angle
        # hand poses (as produced by the estimator) are accepted directly.
        body_model = smplx_lib.create(
            model_path=model_path,
            model_type="smplx",
            gender=gender,
            use_face_contour=False,
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
            num_expression_coeffs=10,
            ext="npz",
        )

        params = dict(
            betas=_as_tensor(smplx_data.get("betas"), (1, 10)),
            global_orient=_as_tensor(smplx_data.get("global_orient"), (1, 3)),
            body_pose=_as_tensor(smplx_data.get("body_pose"), (1, 63)),
            transl=_as_tensor(smplx_data.get("transl"), (1, 3)),
            left_hand_pose=_as_tensor(smplx_data.get("left_hand_pose"), (1, 45)),
            right_hand_pose=_as_tensor(smplx_data.get("right_hand_pose"), (1, 45)),
            jaw_pose=_as_tensor(smplx_data.get("jaw_pose"), (1, 3)),
            leye_pose=_as_tensor(smplx_data.get("leye_pose"), (1, 3)),
            reye_pose=_as_tensor(smplx_data.get("reye_pose"), (1, 3)),
            expression=_as_tensor(smplx_data.get("expression"), (1, 10)),
        )

        with torch.no_grad():
            output = body_model(return_verts=True, **params)

        joints = output.joints[0].detach().cpu().numpy()      # (num_joints, 3)
        vertices = output.vertices[0].detach().cpu().numpy()   # (num_verts, 3)

        # Normalize into Hunyuan3D-Omni space: center on the mesh bbox and
        # scale so the largest dimension fits in [-scale, scale] (mirrors
        # Hunyuan's normalize_mesh). Skeleton uses the same transform so it
        # stays aligned with a mesh normalized the same way.
        bmin = vertices.min(axis=0)
        bmax = vertices.max(axis=0)
        center = (bmax + bmin) / 2.0
        extent = float((bmax - bmin).max())
        if extent <= 0:
            extent = 1.0
        s = (1.0 / extent) * 2.0 * scale

        def norm(p):
            q = (p - center) * s
            if flip_y:
                q = q * np.array([1.0, -1.0, 1.0], dtype=q.dtype)
            if flip_z:
                q = q * np.array([1.0, 1.0, -1.0], dtype=q.dtype)
            return q

        n_joints = joints.shape[0]

        def joint_xyz(name):
            idx = J[name]
            if idx >= n_joints:
                return None
            return norm(joints[idx])

        rows = []
        skeleton_dict = {}
        for bone_name, start_name, end_name in HUNYUAN_BONES:
            start = joint_xyz(start_name)
            end = joint_xyz(end_name)
            # Fall back gracefully if a landmark/tip is missing in this
            # smplx build (e.g. no extra joints): collapse to the start point.
            if start is None:
                start = joint_xyz("pelvis")
            if end is None:
                end = start
            rows.append([float(v) for v in (*start, *end)])
            skeleton_dict[bone_name] = [
                [float(v) for v in start],
                [float(v) for v in end],
            ]

        # [51, 6] text: "x1 y1 z1 x2 y2 z2" per line, np.loadtxt-compatible.
        pose_bone_txt = "\n".join(
            " ".join("%.18e" % v for v in row) for row in rows
        ) + "\n"

        skeleton_bone_json = json.dumps(skeleton_dict, ensure_ascii=False, indent=4)

        saved_path = ""
        if save_name:
            saved_path = self._save(save_name, pose_bone_txt, skeleton_bone_json)

        return (pose_bone_txt, skeleton_bone_json, saved_path)

    def _save(self, save_name, pose_bone_txt, skeleton_bone_json):
        """Write the bone .txt (and companion .json) to ComfyUI's output dir
        when running inside ComfyUI, otherwise to the current directory."""
        try:
            import folder_paths
            out_dir = folder_paths.get_output_directory()
        except Exception:
            out_dir = os.getcwd()

        base = save_name[:-4] if save_name.lower().endswith(".txt") else save_name
        txt_path = os.path.join(out_dir, base + ".txt")
        json_path = os.path.join(out_dir, base + ".json")
        os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
        with open(txt_path, "w") as f:
            f.write(pose_bone_txt)
        with open(json_path, "w") as f:
            f.write(skeleton_bone_json)
        return txt_path


NODE_CLASS_MAPPINGS = {
    "SMPLxExportHunyuanOmni": SMPLxExportHunyuanOmni
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLxExportHunyuanOmni": "SMPLX Export To HunyuanOmni"
}
