# OASIS Brain Analysis 项目自用检查

## 当前定位

- 项目目录已采用较宽泛名称：`oasis_brain_analysis`。
- 当前已经实现并验证的内容仅为 **OASIS 四类脑组织 2D 分割模块**：基线 2D U-Net 与改进 2D U-Net 的公平比较框架。
- 目录名称可容纳后续 3D、VAE、GAN 或其他脑影像分析模块；这些模块目前没有被写成已完成。

## 已有证据

- 真实数据审计：11,328 对 PNG 图像/掩码、354 位受试者、全部 256 x 256；编码 0/85/170/255 已映射为类别 0-3。
- 审计 manifest 与 4 张样例图已生成。
- 自动化测试目前为 18 项。
- GPU smoke run 已证明训练、指标保存和预测面板链路可运行。

## 不能夸大的地方

- smoke run 的 mean Dice 0.0182、mean IoU 0.0094 不是最终效果，不能用于声称模型达到课程目标。
- 尚未完成全数据、固定 manifest 下的 baseline 与 improved 完整对照训练。
- 尚未完成 3-5 个高误差案例的分析，也未完成导师 check-off、PR、README Turnitin PDF 与线上 code review。

## 下一次找导师时带什么

1. 单页 feasibility review（DOCX/PDF）。
2. `evidence/data_audit/dataset_manifest.json` 与样例图。
3. smoke metrics、测试通过记录，以及三个问题：范围、每类 Dice 目标和训练预算。

## 当前最直接结论

工程骨架、数据合规检查和 smoke 证据已经具备；项目尚不能作为完成的第二项目提交。下一项应是导师确认范围，再完成两种模型在同一完整 manifest 上的训练和失败案例证据。
