# 最新 FPN＋U-Net：输入校正、重叠推理和末层适配

用户在上一轮三组负结果后要求“再尝试其他的方法”。本轮独立新增 `tools/fpn_stc_alternatives`，不重复 Memory 温度、全局17偏置、历史软先验、速度缩放或norm伪监督。原trainer、模型、正式ALL4369、bias448与前两轮local入口全部不改。

## 结论

三条方向＋一个固定占据的末层对照都已经真实跑完，没有通过固定16窗筛查，因此**没有扩大64窗，更没有启动服务器全集或修改正式模型**。各行相对各自同次FREE−1参照（约19.425/30.700）：

| 候选 | ΔmIoU | ΔIoU |
|---|---:|---:|
| 历史混淆统计embedding，half | −0.001546 | −0.052999 |
| 历史混淆统计embedding，full | −0.050338 | −0.190619 |
| 重叠128窗口推理 | −0.049213 | −0.020379 |
| 特征条件末层162参数 | −0.800452 | −2.320802 |
| 固定占据、条件语义153参数 | −0.629210 | 0 |

这里只能说明这些固定候选/预算没有提供收益，不证明所有输入适配或微调都无效。162参数loss下降、真实指标下降，与“只让输出层拟合得更好就必然涨指标”不符；153对照说明这次退化也不完全由二元占据范围改变造成。没有据此推导更广泛的网络根因。

## 数据与边界

仍是用户包 `D:/tian/Documents/fpn_stc_local_20261024_204946.zip` 的FULL TRAIN20430×2最终epoch2，2520442参数，源SHA `cedcf7c35123489b0b7e3e0fdca1367241c3baf28c18fb1b8db6e0ff6bef138f`。portable包仅提供真实原64数据，不加载其旧模型。GT future ego、四历史/六未来、STC dense-valid、age0.5、fullgrid1/2/3s不变，无GT visibility输入。

先固定原序前16窗筛查，差的停止；有明确收益的才用相同输出目录扩到64。前16 baseline本来约19.37/30.15，FREE−1约19.43/30.70；**不能把这个16窗分数当全集19+/29+达标**。

embedding/head只用前32场景做有标签目标域适配，不能称无标签/zero-adaptation Camera。原64已经反复开发，后32只是未参与本轮拟合，不是独立封存测试。原模型与缓存只读；hook在退出/异常时清理，最终核对完整tensor fingerprint。完整窗口整数counts落盘，不平均逐窗mIoU。包无Moving实例mask，不能补造该项或用动态语义替代。

## 1. 配对历史语义 embedding 校正

只用前32的四帧**历史**STC/GT语义（不读未来GT拟合），累计`C[STC_class, GT_class]`。计算条件期望embedding `E_new(c)=sum_g P(GT=g|STC=c) E_old(g)`；无支持类别回退原embedding，unknown行不变。固定半强度与全强度两候选，known通道、raw标签、source身份/形状不动。改后的embedding同时服务重新计算的FPN→WM→Memory，非复用旧learned features。

实际32窗历史统计＋16窗筛查约34.3秒。相对同次FREE−1：half为−.001546m/−.052999IoU；full为−.050338/−.190619。均不扩大，不继续扫强度。

## 2. U-Net 重叠小块推理

仅对已算出的Prior与Memory取固定128×128、保留全部Z的2×2重叠窗口。Memory按半XY lattice同步裁剪；每块仍预测六未来，最后用正值高斯权重融合logits。固定sigma=tile/8，不扫窗口/权重。FPN、WM、source motion、完整场景prior不改变。

参考[nnU-Net官方高斯重叠拼接实现](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/sliding_window_prediction.py)的原则，不是复现其完整算法或默认滑窗数。本候选会增加计算，不承诺保留旧FPS；质量筛查耗时不是FPS。

实际16窗19.3秒，FREE−1参照19.425466/30.699954，新19.376252/30.679575，delta−.049213/−.020379。不扩大。

## 3. 末层特征条件校准

区别上一轮只加17个全局偏置：读取冻结U-Net最后8通道特征，用全局共享线性修正 `delta_Z = delta_W X + delta_b`，允许相同类别在不同特征位置有不同调整。只学习18×8+18=**162个标量**，原网络全部冻结。修正独立保存/通过hook应用，不覆盖原卷积权重。

前32未来GT用于拟合（明确有监督适配）；固定Adam lr.02 / 200步 / CE+soft-IoU+L2.01 / 最后一步，不选best。每类/时距/窗至多128点，`类别原体素数/抽样数`带权还原统计；特征均值方差只由这些拟合数据估计。所有候选保留固定FREE−1，不再扫空类bias。

实际32窗拟合＋16筛查约88.3秒。loss1.233768→1.145397，但16窗FREE−1参照19.425311/30.699669，新18.624858/28.378867，delta−.800452/−2.320802。loss下降不是质量提高，不扩大。

### 追加固定二元门的153参数对照

为区分占据范围与语义问题，仅复用上述已采集前32样本（SHA绑定、只读），重拟合17×8+17=153参数的条件语义校准；二元门严格保留原FREE−1，只允许占据体素换语义类别。仍是相同固定200步预算/目标/正则，无新特征提取和后32拟合。

实际拟合＋16窗约46.7秒，参照19.425468/30.700178，新18.796258/30.700178；mIoU−.629210，IoU严格不变。不扩大，不改目标权重或继续扫步数。

## 使用与测试

本地按Windows环境保护技能用 `F:/anaconda3/envs/yoloe/python.exe`，不使用base或安装新包。入口：

`-m tools.fpn_stc_alternatives.run --weights-zip <用户ZIP> --out-dir <新目录> --study embedding|tiled|head|conditional --windows 16`

conditional可加`--fit-samples-from outputs/fpn_stc_alternatives_20261024_head`复用原样本。严格同source/data/实现的输出可加`--resume --windows 64`显式扩大；不自动跑服务器全集或推广模型。

输出 `outputs/fpn_stc_alternatives_20261024_{embedding,tiled,head,conditional}/` 的summary/results/raw integer counts/fit文件；旧结果不覆盖。

4专项测试通过：embedding恒等/无支持回退/原权重不变及异常hook清理；重叠拼接逐点预测守恒/边界覆盖；162与153拟合和分层样本质量/原样本不变；真实U-Net head特征捕获、零patch与单块恒等、多块有限、原model state只读。首轮仅测试夹具tile12大于真实8×8导致拒绝，已将测试尺寸按夹具取4，生产tile128与数值容差未放宽。
