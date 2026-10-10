# 原始 BEV-Planner JSON 的 t0 起点核验

独立只读诊断，不重跑四设置模型，不需 STC、GT occupancy、checkpoint 或 GPU。
一次检查 camera_pred 缓存全部窗口（预计4219），以及这150个场景完整6019行起点。
只使用原始 planner JSON、缓存 identity/六帧位姿、nuScenes identity/ego 元数据。

## 原始文件已找回

本地父目录 `_received/bevplanner_ego_in_bev_with_yaw.json` 存在，大小2,651,124 bytes，
SHA256 `19c04eaf37f531148d5b5719e3cabf8caa35a7140bba9527df47c79aa6783afb`，
与旧 v4 缓存记录完全一致。内容6019行、150场景、每行7×3。
不需要重新下载大数据；上传这一个约2.53MiB的小文件即可。
旧COME网盘链接已被其README标为deprecated，不依赖网络下载作为默认前提。

## 检查什么

- 与旧构建器及 COME 一致：按场景将数字后缀排序，再按完整场景时间/链序号取行。
  例如 `scene-0017-240` 的后缀不是场景内第240帧，禁止把 suffix 直接当 ordinal。
- 原始行第0项的世界XY与实际t0的LIDAR_TOP ego pose比较，容差2cm。
  同时检查其最接近哪一帧、时间偏移，以及前后两行的起点误差。
- 独立重建六位姿：XY为世界坐标，yaw各自相对t0，**不累积**；高度/倾斜继承t0。
  与NPZ缓存六位姿逐元素比较，最大绝对误差容差1e-6；附累积yaw的对照误差。
- 缓存scene/history_tokens/future_tokens/format/tag必须一致；错误身份直接拒绝。
  不读旧BEVStereo历史、latent、未来GT、proposal或mask字段。
- 原始文件SHA严格锁定；场景行数、唯一后缀、完整prev/next/timestamp链必须一致。
  文件和元数据读前后核验；不修改源文件，不覆盖旧报告。

JSON没有sample token。停车/重访位置时多个时刻的XY可能相同，报告
`compatible_ambiguous_xy`，不能仅凭起点坐标宣称唯一帧身份已证明。
`compatible_unique_xy` 表示排序、起点几何、缓存重建三者兼容，仍不是JSON内不存在的token证明。
不匹配只是诊断发现：可能有行错位、不同传感器时刻/位姿来源或转换差异，
不能自动换成“更近的行”、调整planner或用未来GT对齐正式评分。

## 服务器命令

先将本地JSON上传到 `/root/nas/occ/swfm/data/come_protocol/bevplanner_ego_in_bev_with_yaw.json`。
没有此目录可先执行 `mkdir -p /root/nas/occ/swfm/data/come_protocol`。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/real_motion/run_p0_f9_original_planner_t0_audit.sh
```

自定义位置可设 `STC_PLANNER_JSON=/actual/path/file.json`、`STC_PLAN_CACHE`、`DATAROOT`。
新输出为 `outputs/p0_f9_joint_surface_ccr/planner_origin_all_时间_PID/summary.txt` 和
`evaluation.json`。只需发回终端打印的摘要；完整报告保留逐窗/错位详情。
若原始文件缺失脚本明确停止，不会冒充“核验通过”或启动模型重评。

旧 STC four/shared/branch evaluator 和协议文件完全不改，原运行/续评指纹不变。
本地测试仅能验证文件SHA、格式及合成错位/歧义案例；真实t0对应结果仍需服务器的nuScenes元数据和缓存。
