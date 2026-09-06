import json
import numpy as np

# Hunyuan3D‑Omni官方52根骨骼配对，顺序严禁改动
OFFICIAL_BONE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (1, 6), (6, 7), (7, 8), (8, 9),
    (1,10),(10,11),(11,12),(12,13),
    (0,14),(14,15),(15,16),(16,17),
    (0,18),(18,19),(19,20),(20,21),
    (7,22),(22,23),(23,24),
    (7,25),(25,26),(26,27),
    (7,28),(28,29),(29,30),
    (7,31),(31,32),(32,33),
    (7,34),(34,35),(35,36),
    (11,37),(37,38),(38,39),
    (11,40),(40,41),(41,42),
    (11,43),(43,44),(44,45),
    (11,46),(46,47),(47,48),
    (11,49),(49,50),(50,51)
]

class SMPLxExportHunyuanOmni:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smplx": ("SMPLX",),
            }
        }
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("pose_bone_txt","skeleton_post_json")
    FUNCTION = "export"
    CATEGORY = "SMPLx-Estimator/Export"

    def export(self, smplx):
        # 直接读取插件已经计算好的关节坐标，不需要smplx库和模型前向传播
        joints_tensor = smplx["joints"][0]  # [54,3]
        joints_np = joints_tensor.detach().cpu().numpy()

        # =====生成PoseMaster txt（包围盒归一化，解决人物漂移）=====
        all_points = joints_np.copy()
        center = (all_points.max(axis=0)+all_points.min(axis=0)) / 2.0
        scale = (all_points.max(axis=0)-all_points.min(axis=0)).max()
        joints_norm = (joints_np - center) / scale

        txt_lines = []
        for p_idx,c_idx in OFFICIAL_BONE_PAIRS:
            start = joints_norm[p_idx]
            end = joints_norm[c_idx]
            dir_vec = end - start
            row = [*start.tolist(), *dir_vec.tolist()]
            line = " ".join(f"{v:.6f}" for v in row)
            txt_lines.append(line)
        pose_txt_content = "\n".join(txt_lines)

        # =====生成post.json skeleton格式=====
        joints_list = joints_np.tolist()
        json_data = {
            "control_type":"skeleton",
            "skeleton":{
                "joints_3d":joints_list
            }
        }
        json_content = json.dumps(json_data,indent=2,ensure_ascii=False)
        return (pose_txt_content,json_content,)

NODE_CLASS_MAPPINGS = {
    "SMPLxExportHunyuanOmni":SMPLxExportHunyuanOmni
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLxExportHunyuanOmni":"Export→Hunyuan3D‑Omni(TXT+JSON)"
}
