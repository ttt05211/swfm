# 固定终版候选：旧组合＋192 norm；单卡2/4进程全集

## 科学配置

本入口只整合用户明确指定的版本：**clean1000主体＋half-pair/LS KTA＋历史年龄记忆＋历史静态软支持＋既有后验校准＋固定200步的192个Refiner norm参数**。

没有加入静态配准、动态补形状、其他小头、模拟噪声、双模型融合或新动作阈值。不重新训练、重新拟合校准，不使用最新dense随机主体或内部ego头。

源主体 SHA256 固定 `17d649046fe7eafdbd1ce08fb8350bb90c93a757faeccdd544266113070aea44`，由原服务器导出的小ZIP中的 clean1000 加载；原网络和checkpoint不覆盖。原ZIP继续复用，不需要下载新大文件。

`tools/stc_norm_final/norm192.json` 随代码提供六个具名FP32参数数组（总192个数），从既有 `next_norm_matched200/norm_final.pt` 的 **target** 臂准确导出，不从GT打分重新拟合。附源SHA、原小checkpoint SHA、固定200步训练配方及历史输入32窗键。`posterior_fit.json` 是既有固定校准，非重新拟合。

目标32场景的历史STC一致性用于此前的无标签参数适配；未来GT占据没有参与该臂拟合。GT future ego 是这次STC+GT任务明示条件。后验校准也使用未标注目标输入。故必须声明 **unlabeled target-domain adaptation / transductive development**，不是完全target-information-free或独立测试。开发64反复检查的事实不因这次全集评估消失。

## 全集与对照

- 完整 VAL4369 / 150场景，不按planner覆盖率裁剪。
- 四历史STC dense-valid，无GT visibility mask；GT future ego；全部六帧预测后才读未来占据评分。
- 报1/2/3秒及平均，full-grid标准IoU/mIoU；Moving本入口不评，与上一完整组合报告一致。
- 三路：`shared/transport`、`previous/semantic_refine`、`norm192/semantic_refine`。
- 新旧两路共享历史编码、WM、未来geometry/prior/memory，仅分别读取Refiner；每次切换norm参数后准确恢复原值，任务结束核验源权重fingerprint。
- 这是固定方法全集验证，不按结果选择checkpoint、阈值或方法。不能保证旧64的小提升在全集复现。

## 单卡多进程实现

只允许一张可见GPU，例如 `NORM_FINAL_GPU=0`。所有子进程继承同一个 `CUDA_VISIBLE_DEVICES=0`，内部各用cuda:0；没有GPU1路径。

支持1（本地小显存验证）、2、4进程。完整有序键按 `keys[rank::processes]` 无padding分片，各自保存完整窗口前缀与原始整数confusion。父进程核验分片身份／GT边际，求和整数计数后算分，**不平均各进程的mIoU**。

新独立目录和协议，不导入旧H1/H6结果充新norm分数；旧评估、源checkpoint、旧cache、旧四指纹文件保持不变。每8个完整窗口落盘，SIGINT/SIGTERM由本任务自己创建的子进程处理，完整窗口后保存；不会扫描或停止其他任务。突发进程崩溃最多重算尚未保存的本分片窗口。

每个输出绑定完整population、固定资产SHA、实现文件SHA、源body／bundle manifest SHA、数据metadata SHA、进程数和H分组。恢复不得偷偷改变2/4进程或模型／资产。输出与原训练目录分离。

## 2进程／4进程测速

默认同一前32窗、同一固定方法、H6，按 **2→4→4→2** 顺序各跑一次。每worker先做2窗预热，再经共同起点barrier开始计时；预热不加入评分/计时窗口。保留整个启动墙时和steady吞吐，计时包含历史I/O、几何准备、新旧两路读出及指标；这是评估吞吐，**不是旧模型FPS**。

只在4进程两轮都快于对应2进程时建议4；否则2。2必须两次完成。4进程OOM或其他失败原样留FAIL／日志，不改变模型、H6或容差；只要2两轮完成可建议2。两种执行的评分差异也记录，不声称BF16 byte identity，不根据质量挑进程数。

不要同时跑其他GPU训练／评估后再据此判断进程数，否则竞争会污染测速。本入口不会擅自停止其他进程。

## 服务器一条后台工作流

先在服务器OccFM环境拉取本分支。已有 `STC_REPLAY_BUNDLE` 可保留；未设置时自动定位既有 `outputs/p0_f9_shared_history/stc_overnight_bundle_*/stc_overnight_bundle.zip`。原dataroot、metadata与STC compact路径默认沿用旧入口；必要时用 `DATAROOT`、`DEV_CACHE`、`DEV_INFO`、`STC_ROOT` 指定。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export NORM_FINAL_GPU=0
export NORM_WORKFLOW_DIR="$PWD/outputs/p0_f9_shared_history/stc_norm192_final_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/stc_norm_final/workflow.sh > "${NORM_WORKFLOW_DIR}.log" 2>&1 &
tail -F "${NORM_WORKFLOW_DIR}.log"
```

此工作流先小范围2/4测速，再仅按速度建议自动选择2或4，运行完整4369。测速通常比全集短得多，但不承诺具体L40S加速倍数。

查看结果：

```bash
cat "$NORM_WORKFLOW_DIR/speed/summary.txt"
cat "$NORM_WORKFLOW_DIR/all/summary.txt"
```

更换终端后不知道路径时：

```bash
export NORM_WORKFLOW_DIR="$(find "$PWD/outputs/p0_f9_shared_history" -maxdepth 1 -type d -name 'stc_norm192_final_*' | sort | tail -1)"
tail -F "${NORM_WORKFLOW_DIR}.log"
```

单独直接跑4进程（跳过测速，不是默认推荐）：

```bash
export NORM_FINAL_DIR="$PWD/outputs/p0_f9_shared_history/stc_norm192_4proc_$(date +%Y%m%d_%H%M%S)"
export NORM_FINAL_WORKERS=4
bash tools/stc_norm_final/run.sh all
```

## 仅恢复已开始的全集，不重测速度

```bash
export NORM_FINAL_GPU=0
export NORM_FINAL_DIR="$NORM_WORKFLOW_DIR/all"
export NORM_SPEED_REPORT="$NORM_WORKFLOW_DIR/speed"
export NORM_FINAL_WORKERS=auto
export NORM_FINAL_RESUME=1
nohup bash tools/stc_norm_final/run.sh all > "$NORM_WORKFLOW_DIR/resume.log" 2>&1 &
tail -F "$NORM_WORKFLOW_DIR/resume.log"
```

必须是已有原目录；原进程数由既有speed建议确定，不可中途换2/4继续混计数。若测速阶段中断，保留失败输出，重新指定新WORKFLOW目录运行，不覆盖旧目录。

## 本地证据与边界

- 新入口真实CUDA64窗 H1 complete：旧组合 **18.508201 / 28.760883**，norm192 **18.525690 / 29.087418**，与先前独立本地对照一致（仅微小BF16重复差异）。该轮执行后加了仅probe barrier／报告文案，不改预测算式；源记录保留不作为新版恢复输入。
- 修订入口真实单GPU双进程H1／8窗、预热barrier联调 complete，完整新旧预测与整数汇总正常；8窗不是质量选择依据。
- 专项加旧 parallel／replay／verifier／boundary 共 **30 passed in 8.88s**。覆盖仅192参数变化、原参数准确恢复、2/4同GPU分配、11窗不等分合并、错误分片拒绝、目标读取时序、完整恢复不重算、资产变化拒绝、4仅双轮更快才选等。
- 两个Bash入口语法检查通过。RTX3050 Laptop仅4GiB，不冒称本地4进程H6实际性能；真实2/4吞吐交由服务器probe测出。
- 没有训练/代启动服务器、自动质量推广或新FPS结论。
