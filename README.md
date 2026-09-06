# SMPLx Export → Hunyuan3D-Omni

一个 ComfyUI 自定义节点插件，将 [ComfyUI-SMPLx-Estimator](https://github.com/ameliacode/ComfyUI-SMPLx-Estimator) 输出的 `SMPLX` 姿态数据，转换为 [Hunyuan3D-Omni](https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni) 姿态控制（`--control_type pose`）所需的骨骼文件。

## 原理

- **输入**：`SMPLX` 字典，包含 `global_orient (1,3)`、`body_pose (1,63)`、`betas (1,10)`、
  `transl (1,3)`、`left_hand_pose (1,45)`、`right_hand_pose (1,45)`、`jaw_pose (1,3)`、
  `expression (1,10)` 等（即 MultiHMR 节点的输出）。
- 节点用 `smplx` 库做一次前向计算得到 55+ 个关节的三维坐标，
  再按 **Hunyuan3D-Omni 的 51 根骨骼定义** 组装成骨架。
- **归一化**：与 Hunyuan 的 `normalize_mesh` 一致——以网格包围盒中心归零、最长边缩放到
  `[-scale, scale]`（默认 `scale=0.9999`，Y 轴朝上）。这样骨架与同样归一化的网格对齐。

## 输出

Hunyuan3D-Omni 的 `infer_pose` 用 `np.loadtxt` 读取一个 `[51, 6]` 的纯文本文件，
每行是一根骨骼的起止点 `x1 y1 z1 x2 y2 z2`。本节点输出：

- `pose_bone_txt`：`[51, 6]` 骨骼文本，可直接保存为 `xxx_bone.txt` 交给 Hunyuan3D-Omni。
- `skeleton_bone_json`：与官方 `*_bone.json` 同格式，`{ "Bone_Name": [[x1,y1,z1],[x2,y2,z2]] }`。
- `saved_path`：当填写了 `save_name` 时，落盘后的 `.txt` 路径（同时写出同名 `.json`）。

## 参数

| 参数 | 说明 |
| --- | --- |
| `smplx` | SMPLX 输入 |
| `model_path` | SMPLX 模型目录（默认 `models/smplx`） |
| `gender` | `neutral` / `male` / `female` |
| `scale` | 归一化目标半边长（默认 `0.9999`） |
| `flip_y` / `flip_z` | 需要时翻转坐标轴以对齐坐标系 |
| `save_name` | 非空时写出骨骼文件到 ComfyUI 输出目录 |

## 在 Hunyuan3D-Omni 中使用

把导出的 `xxx_bone.txt` 填入 `inference.py` 的 `pose_configs`：

```python
pose_configs = {
    "my_pose": "/path/to/xxx_bone.txt",
}
```

然后运行：

```bash
python3 inference.py --control_type pose
```

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jyworker/smplx_export_hunyuan_omni.git
pip install -r smplx_export_hunyuan_omni/requirements.txt
```

将 SMPLX 模型放到 `models/smplx` 目录，重启 ComfyUI。

## 节点

| 名称 | 分类 | 输入 | 输出 |
| --- | --- | --- | --- |
| SMPLX Export To HunyuanOmni | SMPLx | `SMPLX` | `pose_bone_txt`, `skeleton_bone_json`, `saved_path` |
