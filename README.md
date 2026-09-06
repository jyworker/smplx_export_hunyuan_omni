# SMPLx Export → Hunyuan3D-Omni

一个 ComfyUI 自定义节点插件，将 SMPLX 姿态数据导出为 Hunyuan3D-Omni 所需的骨骼格式。

## 功能

节点 **Export→Hunyuan3D-Omni(TXT+JSON)** 接收一个 `SMPLX` 输入，输出：

- `pose_bone_txt`：PoseMaster 使用的 `pose_bone.txt` 内容（52 根官方骨骼，带归一化以防漂移）
- `skeleton_post_json`：`skeleton_post.json` 内容（54 个关节的三维坐标）

## 安装

1. 将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/jyworker/smplx_export_hunyuan_omni.git
   ```

2. 安装依赖：

   ```bash
   pip install -r smplx_export_hunyuan_omni/requirements.txt
   ```

3. 将 SMPLX neutral 模型放到 `models/smplx` 目录下。

4. 重启 ComfyUI。

## 节点

| 名称 | 分类 | 输入 | 输出 |
| --- | --- | --- | --- |
| Export→Hunyuan3D-Omni(TXT+JSON) | SMPLx-Estimator/Export | `SMPLX` | `pose_bone_txt`, `skeleton_post_json` |
