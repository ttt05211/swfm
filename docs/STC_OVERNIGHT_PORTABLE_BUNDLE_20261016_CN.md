# STC 夜间研究数据包

目标：在原完整4369窗、full-grid、GT future ego的STC输入协议上同时超过18.97 mIoU /28.77 IoU；未有真实完整验证不得声称达成。

当前本地仅RTX3050 Laptop4GiB和测试数据，缺最新真实checkpoint/标签，无远程执行连接。用户明确先打包下载到本地，再继续研究。新增CPU-only export，不改原训练/评估文件、权重或缓存。

默认包包含：

- 原dense epoch2（必须），以及能找到的clean1000、旧正式GTepoch2。只导出模型权重、configs、来源契约/SHA，不含Adam/RNG。
- 完整VAL4369所有必要GT历史/未来labels.npz、STC四历史labels.npz，按scene/token去重；另64个原TRAIN干净GT窗口。
- 每窗口history/future tokens、实际四历史时间、4/6 poses、PrepareConfig。无需nuScenes SDK完整JSON、原始相机、点云、旧特征bank/E14文件。
- 原reference64及其整个场景不进入stc_fit/stc_selection。其他VAL场景固定SHA分区，仅为后续研究准备数据，不等于已授权把VAL监督训练后冒充标准全集成绩。

重要：若使用VAL未来GT监督拟合神经网络，必须只在未参与拟合的整个场景评估，不能把含训练场景的4369分数称标准VAL提升。标准全集验证路线只允许原TRAIN监督、因果无标签适配及透明开发集模型选择；原64也已经反复查看，并非独立测试。

服务器命令：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2 \
  && git merge --ff-only FETCH_HEAD
bash tools/stc_overnight/export.sh
```

如多源可设置DENSE_DDP_DIR或CLEAN1000_CKPT。默认dense选最近已完成epoch2，clean1000选最近stc_clean_continue1000_*/train/last.pt；原GT2固定旧正式路径。缺dense报错，缺两个可选模型明确打印OPTIONAL_MODEL_MISSING。

脚本在写包前打印去重帧数及实际bytes。只在新输出目录写manifest/模型和一个stc_overnight_bundle.zip；NPZ直接从原路径写ZIP_STORED，不额外复制一套标签。完成打印DOWNLOAD_THIS_FILE绝对路径及receipt大小/SHA。下载那个ZIP即可，内部已有manifest和模型；无需单独下载目录或receipt。

建议保存到本地 `_received/stc_overnight_bundle.zip`，完整绝对路径位于stochocc-diagnosis根目录下。导出原文件前后不变，小包往返及完整人口/场景划分2项测试通过；未在服务器真实打包，大小由实际文件决定，不预先承诺。
