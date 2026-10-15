# Segmentation: Anatomical Contouring and Radiotherapy Planning 中期检查准备

## 当前项目定位

- 当前项目是 COMP3710 的 Hard 难度路线：HipMRI 3D 前列腺与相关器官分割。
- 比较对象是 plain 3D U-Net baseline 与 3D Improved U-Net；改进模型含 volumetric residual units 和 deep supervision。
- 2D 阶段是本项目的工程起点，但当前报告不再声称 OASIS 2D 是最终项目，也不复用 OASIS 的数据、指标或结果。

## 研究问题

放疗勾画需要连续且可审核的 3D 边界。问题不是只比较平均 Dice，而是检验 3D 上下文、residual units 与 deep supervision 是否能减少 prostate base、apex、跨切片不连续，以及膀胱或直肠边界泄漏；同时确认收益是否值得额外显存和推理时间。

## 当前可验证证据

- `dataset.py` 已有递归 NIfTI image-mask 配对、3D shape/affine、有限强度及整数标签检查。
- patient-level seed-42 train/validation/test manifest 有零重叠验证；不会按 slice 或 patch 随机切分。
- `BaselineUNet3D` 与 `ImprovedUNet3D` 已实现；改进模型训练时输出两个 deep-supervision heads，评估时只输出最终 segmentation logits。
- 8 项自动化测试通过：NIfTI 配对与拒绝异常数据、病人切分、3D forward/backward、deep supervision loss、held-out metrics 和 validation-only checkpoint selection。
- `COMP3710_HipMRI_Hard_Feasibility_Review.pdf` 是一页，已明确 Hard、HipMRI 3D、prostate Dice >= 0.70、资源约束与 fallback。

## 不得夸大的内容

- 未在真实 HipMRI 数据上完成 audit；不可声称已确认患者数、类别编码、体积几何或 affine 实测结果。
- 未完成 Rangpur GPU smoke run、full training、真实 held-out Dice、VRAM、per-case inference time 或 failure panels。
- 目前没有 tutor check-off 记录、课程 PR、Turnitin receipt 或 teaching-team review response。
- 本地单元测试证明代码合同，不证明临床可用性或课程目标已经达到。

## 会前清单

- [ ] 打开一页 Hard 3D feasibility PDF，确认标题、模型、Dice target 和风险段落。
- [ ] 打开 `topic-recognition` 分支及渐进式 commit history。
- [ ] 展示 `dataset.py`、`modules.py`、`train.py` 和 8 项测试通过记录。
- [ ] 准备说明：真实 HipMRI audit 和 GPU 训练仍是下一阶段，而不是已完成证据。
- [ ] 记录 tutor 对 patch size、batch size、label encoding、background 统计口径和计算预算的要求。

## 约三分钟讲稿

> 我的项目是 Segmentation: Anatomical Contouring and Radiotherapy Planning，选择 Hard 路线的 HipMRI 3D segmentation。我会以 plain 3D U-Net 为 baseline，并比较使用 volumetric residual units 和 deep supervision 的 3D Improved U-Net。
>
> 目标不是只提高平均 Dice，而是判断 3D 上下文是否改善 prostate base、apex、跨切片连续性和邻近器官边界。数据会按 patient 而不是 slice 切分，两个模型必须使用同一 saved manifest。
>
> 当前已完成 NIfTI 配对和 shape/affine 检查、patient-level split、3D 模型前反向传播、deep supervision loss 和 held-out metrics 的测试。真实 HipMRI audit、Rangpur smoke run、完整训练和 Dice 结果尚未完成，因此我不会把本地测试说成最终模型效果。
>
> 我希望确认实际 label encoding、Dice 是否排除 background、合理的 patch/batch 配置，以及中期后完整训练要保留哪些日志和 failure evidence。

## 必问问题

1. HipMRI 3D Improved U-Net 的 Hard scope、prostate Dice >= 0.70 与 patient-level split 是否符合课程预期？
2. prostate label 的实际编码是什么，background 是否单独报告但不计入 mean Dice？
3. 是否建议特定 patch size、batch size、mixed precision、epoch budget，及 base/apex failure panels 的最低要求？

## 导师记录模板

- Practical 日期：
- 导师：
- 确认或修改后的 scope：
- label encoding / Dice 口径：
- patch、batch、epoch 或资源限制：
- 必须补充的证据：
- Check-off 是否已记录：是 / 否
- 下一步：
