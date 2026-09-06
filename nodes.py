import smplx
import torch
import json


class SMPLxExportHunyuanOmni:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smplx": ("SMPLX",),
                "model_path": ("STRING", {"default": "models/smplx"}),
                "gender": (["neutral", "male", "female"], {"default": "neutral"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pose_bone_txt", "skeleton_post_json")
    FUNCTION = "export"
    CATEGORY = "SMPLx"

    def export(self, smplx_data, model_path, gender):
        # load smplx model
        smpl_model = smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender=gender,
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext="npz"
        )

        betas = smplx_data.get("betas", torch.zeros([1, 10]))
        body_pose = smplx_data.get("body_pose", torch.zeros([1, 21, 3]))
        global_orient = smplx_data.get("global_orient", torch.zeros([1, 3]))
        transl = smplx_data.get("transl", torch.zeros([1, 3]))
        left_hand_pose = smplx_data.get("left_hand_pose", torch.zeros([1, 15, 3]))
        right_hand_pose = smplx_data.get("right_hand_pose", torch.zeros([1, 15, 3]))
        expression = smplx_data.get("expression", torch.zeros([1, 10]))

        output = smpl_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            expression=expression,
            return_verts=True
        )

        joints = output.joints.detach().cpu().numpy()
        vertices = output.vertices.detach().cpu().numpy()

        # build skeleton json for Hunyuan3D Omni
        skeleton_dict = {
            "joints": joints.tolist(),
            "verts": vertices.tolist(),
            "betas": betas.detach().cpu().numpy().tolist(),
            "body_pose": body_pose.detach().cpu().numpy().tolist(),
            "global_orient": global_orient.detach().cpu().numpy().tolist(),
            "transl": transl.detach().cpu().numpy().tolist(),
        }
        skeleton_post_json = json.dumps(skeleton_dict, ensure_ascii=False, indent=2)

        # simple pose bone text
        pose_bone_txt = f"""global_orient={global_orient.tolist()}
transl={transl.tolist()}
body_pose={body_pose.shape}
betas={betas.tolist()}
"""
        return (pose_bone_txt, skeleton_post_json)


NODE_CLASS_MAPPINGS = {
    "SMPLxExportHunyuanOmni": SMPLxExportHunyuanOmni
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLxExportHunyuanOmni": "SMPLX Export To HunyuanOmni"
}
