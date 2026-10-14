# OASIS Brain Analysis 中期评审自用检查

## 当前定位

- 项目目录为 `oasis_brain_analysis`，便于后续增加其他 OASIS 脑影像分析模块。
- 当前实际实现并验证的内容只有 **OASIS 四类脑组织 2D 分割模块**：基线 2D U-Net 与改进 2D U-Net 的公平比较框架。
- 3D、VAE、GAN 或其他扩展尚未实现，面谈时不应说成已完成。

## 项目要解决什么问题

- 输入是一张预处理后的灰度脑部 MRI slice，输出是每个像素所属的组织类别。
- 本项目比较两个网络：简单的 baseline 2D U-Net，和加入 residual block、attention gate、bottleneck dropout 的 improved 2D U-Net。
- 比较必须公平：两种模型使用同一数据、同一 seed、同一个 subject-disjoint split、同一评价方式。
- 研究问题不是“能不能运行模型”，而是：改进结构能否在每个组织类别上带来足够的改善，且改善是否值得额外参数、显存和推理成本。

## 数据与切分要能讲清楚

- 审计确认有 11,328 对 PNG 图像/掩码，来自 354 位受试者；每对都是 256 x 256。
- 图像文件名以 `case_` 开头，mask 以 `seg_` 开头；加载器递归搜索嵌套目录并配对。
- mask 原始显示编码为 0、85、170、255；训练前映射为类别 0、1、2、3。不能把 85、170、255 直接送进 cross-entropy。
- 切分单位是受试者而不是单张 slice。同一受试者的 slice 不能同时出现在训练集和测试集，否则测试指标会偏高且没有真实泛化意义。
- 当前固定策略为 seed 42 的 subject-disjoint split。完整实验时 baseline 和 improved 必须共用保存下来的 manifest。

## 模型与指标要能讲清楚

- baseline 2D U-Net 使用编码器、池化、解码器、转置卷积和直接 skip connection。
- improved 2D U-Net 的 residual block 帮助特征传播；attention gate 过滤 skip feature；dropout 用于降低 bottleneck 过拟合风险。
- Dice 衡量预测区域和真实区域的重叠；IoU 衡量交集占并集。两者都需要按类别报告，不能只看平均值。
- 还要记录参数量、peak VRAM、总推理时间和单案例时间。模型指标稍高但资源成本明显过大，也未必值得采用。
- FP 是预测为该类别但真实不是；FN 是真实属于该类别但模型漏掉。高误差案例应说明病例、类别、FP/FN 模式和可能原因。

## 已有、可当场展示的证据

- 数据审计：11,328 对 PNG 图像/掩码、354 位受试者、全部 256 x 256；编码 0/85/170/255 已映射为类别 0-3。
- `data_audit/dataset_manifest.json` 与 4 张确定性样例图。
- 18 项自动化测试通过。
- GPU smoke run 已证明训练、指标 JSON、checkpoint 和预测面板链路可运行。

## smoke run 的正确说法

- 已验证：数据可以被读取；标签能映射；模型能前向和反向传播；训练可保存 checkpoint；测试可写出 metrics JSON；预测可生成面板。
- 未验证：模型在完整数据上达到课程目标；improved 优于 baseline；所有类别都有可接受的 Dice；资源预算适合完整实验。
- 当前 smoke run 使用 baseline、1 epoch、每个 split 仅 8 个案例。mean Dice 0.0182、mean IoU 0.0094 只说明小型流程没有中断，不是可以展示为模型质量的成绩。

## 不能夸大的地方

- smoke run 的 mean Dice 0.0182、mean IoU 0.0094 只是极小训练的流程证据，不是最终效果。
- 尚未完成完整固定 manifest 下 baseline 与 improved 的对照训练。
- 尚未完成每类指标和资源比较、3-5 个高误差案例、导师 check-off、PR、README Turnitin PDF 与线上 code review。

## 会前检查清单

- [ ] 打开单页 feasibility review 的 PDF，确认标题为 OASIS Brain Analysis，且只有一页。
- [ ] 打开 GitHub 的 `topic-recognition` 分支和渐进式提交记录。
- [ ] 打开 `data_audit/dataset_manifest.json`、4 张样例图、测试通过记录和 smoke metrics。
- [ ] 准备充电器；不需要 slides、完整训练结果或 live demo。
- [ ] 带着下方三个问题，记录导师给出的具体要求。

## 约 3 分钟讲稿

> 我的项目目前实现的是 OASIS 四类脑组织 2D 分割。我会在相同的 subject-disjoint split 上比较 baseline 2D U-Net 和加入 residual、attention gate、dropout 的 improved 2D U-Net。
>
> 真实数据审计确认有 11,328 对图像/掩码、354 位受试者，全部为 256 x 256。mask 的 0、85、170、255 编码已映射为四个训练类别。切分按 subject 完成，因此同一受试者不会同时出现在训练和测试中。
>
> 我会比较每类 Dice/IoU、参数量、VRAM、推理时间，并检查 3-5 个高误差 FP/FN 案例。当前 smoke run 仅证明数据、训练、指标保存和预测面板流程可运行；低 Dice 不是最终模型结论。
>
> 我希望确认当前 Easy scope、每类 Dice 的报告口径，以及完整训练的预算和中期需要保留的证据。

## 导师可能追问与回答方向

| 可能问题 | 回答重点 |
|---|---|
| 为什么使用 2D？ | 当前是可控的课程原型。先完成可复现的 2D 公平比较，不声称已经做了 3D。 |
| 如何避免 data leakage？ | 以 subject 为单位切分，不以 slice 随机切分；同一 subject 不会跨 train、validation、test。 |
| 为什么不能只看 mean Dice？ | 类别不平衡会掩盖小组织或边界失败；必须提供每类 Dice/IoU 和 FP/FN 案例。 |
| 当前分数为什么低？ | 当前是 1 epoch、8 cases/split 的 smoke-run，目的是验证工程链路，不是模型评估。 |
| improved 如何证明值得？ | 同一 manifest 下比较每类指标、参数、VRAM、推理时间和失败案例，不凭单一平均分下结论。 |

## 必问的三个问题

1. 当前的 OASIS 四类 2D segmentation scope 是否符合 Easy 难度要求？
2. Dice 是否要求每个前景类别单独达到目标？background 应怎样报告？
3. baseline 与 improved 的完整训练，您希望使用多少 epochs、哪些资源记录和哪些中期证据？

## 导师回答记录模板

- 日期和 practical：
- 导师姓名或角色：
- 确认的项目范围：
- 指标口径和 threshold：
- 训练 epochs、batch size 或资源限制：
- 必须补充的证据：
- 是否完成 check-off 记录：是 / 否
- 下一步与截止时间：

## 会后下一步

1. 将导师确认内容写入 README 和本文件。
2. 固定 manifest，依次运行 baseline 与 improved 的完整训练。
3. 保存 checkpoint、history、test metrics、资源记录和 3-5 个失败案例。
4. 更新 README 后创建或更新课程要求的 PR，再处理 Turnitin PDF 与 code review。

## 面谈结束前核对

- [ ] 导师已明确确认或修改范围。
- [ ] 已问清 background 是否计入平均 Dice。
- [ ] 已问清完整训练的 epoch、batch size 或资源上限。
- [ ] 已确认需要保存的结果文件和 failure panels 数量。
- [ ] 已确认 practical check-off 是否已被记录。
