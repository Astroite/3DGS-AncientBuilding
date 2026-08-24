# 009 首跑 3DGS 质量审核意见

- 审核日期：2026-08-24
- 地点：`yanguan-ancient-town-20260822`
- 场景：`night-walk-4k`
- 运行：`20260824T022046Z-5679786b`
- 配置哈希：`5679786b71621b06191101aaa50fe27dd83d215193b6532ba7cad6eb12d62323`
- 导出版本：`exports/v001`
- 审核对象：重建与训练配置的质量瓶颈
- 审核性质：**技术复盘，不是 QA 裁决。** 该运行状态保持 `needs_review`；接受或拒绝仍由 `gsdb qa approve/reject` 人工决定。

## 结论

流程闭环是成立的，不需要推翻。注册率 99.63%、单一连通模型、相机高度沿重力方向仅变化 0.173 单位、未触发降级重建，说明 COLMAP 与 Splatfacto 的衔接、遮罩注入和全局几何都正常工作。

画质差的原因不在流程结构，而在三处可量化的资源浪费，加上一个素材层面的硬上限：

1. 投影阶段丢掉了源素材一半以上的角分辨率（最确定、最易修）。
2. 训练严重不足，GPU 大部分算力闲置，高斯数偏低一个量级。
3. COLMAP 把精确已知的内参交给 BA 去 refine，并把同一全景的 8 个共心视图当作 8 个独立相机。
4. 素材本身是 4K 档夜景单次前向行走，存在无法通过后处理突破的上限。

以下每一项都给出实测依据、判断理由和具体改动位置。

## 审核依据与复现方法

本审核不依赖主观观感，主要证据取自可版本控制的产物：

| 证据来源 | 用途 |
| --- | --- |
| `scenes/night-walk-4k/exports/v001/transforms.json` | 内参、2152 个注册位姿、轨迹几何 |
| `scenes/night-walk-4k/exports/v001/splat.ply`（头部） | 高斯数量 |
| `scenes/night-walk-4k/runs/20260824T022046Z-5679786b.yaml` | 各阶段耗时、显存/磁盘峰值、遮罩与重建指标 |
| `scenes/night-walk-4k/captures/capture-009-4k.yaml` | 相机型号、镜头流分辨率、码率、导出设置 |

轨迹类结论由 `transforms.json` 直接计算：按 `frame_NNNNNN_V` 文件名把 2152 个视图归到 270 个全景，取每个全景 8 个视图平移分量的均值作为该帧光心；重力方向取每帧 8 个视线方向（`-z` 列）最佳拟合平面的法向。COLMAP 单位为任意尺度，涉及米制的换算均在文中标注为估算。

`work/<RunId>/` 已按设计被 Git 忽略，本次审核未依赖其中的中间文件；因此以上结论在任何一份仓库副本上均可重算。

## 一、投影阶段丢掉一半以上的角分辨率

### 实测

| 项 | 实测值 |
| --- | --- |
| 透视图尺寸 | 960 × 960 |
| `fl_x` / `fl_y` | 276.9756 / 277.1690 |
| 由焦距反推 FOV | 120°（与 `projection_fov_degrees` 一致） |
| 视图**中心**角分辨率 | 277 px/rad = **4.84 px/度** |
| 源全景角分辨率 | 3840 / 360 = **10.67 px/度** |
| 中心欠采样倍数 | **2.21×** |

透视投影的局部尺度在光轴处最低、随离轴角以 `1/cos²θ` 增长。因此当前配置在每张图**最有用的中心区域欠采样 2.2 倍**，却在边缘过采样。COLMAP 的 SIFT 和 Splatfacto 的像素监督都只能看到已经糊掉的输入，后续任何环节都无法恢复。

### 原因

`src/gsdb/reconstruction.py` 的 `project_equirectangular_frames` 使用了 nerfstudio 的启发式 `compute_resolution_from_equirect`，它对 8 视图返回 960²，与实际使用的 120° FOV 无关。`resolution` 本身已经是 `generate_planar_projections_from_equirectangular` 的显式参数，替换成本很低。

### 建议

按源全景的角密度反推视图边长，使视图中心不欠采样：

```python
def projection_resolution(equirect_width: int, fov_degrees: float) -> tuple[int, int]:
    """按源全景的角密度给透视图定尺寸，使视图中心不欠采样。"""
    focal = equirect_width / (2.0 * math.pi)  # 源图每弧度像素数
    side = round(2.0 * focal * math.tan(math.radians(fov_degrees) / 2.0))
    return (side, side)
```

对 3840 宽、120° FOV 得 2117²。可选档位：

| FOV | 无欠采样所需边长 | 相对像素量 | 垂直覆盖 |
| ---: | ---: | ---: | ---: |
| 120°（当前） | 2117 | 4.9× | ±60° |
| 110° | 1746 | 3.3× | ±55° |
| 100° | 1456 | 2.3× | ±50° |
| 90° | 1222 | 1.6× | ±45° |

**不建议为省算力把 FOV 压到 90°。** 视图为正方形，垂直 FOV 等于水平 FOV；压到 90° 会丢掉 ±45° 以上的天区，正是屋檐、翼角和屋脊所在。建议保持 120°、边长取 2048，算力由第四节释放。

同时把 `projection_fov_degrees` 中硬编码的 `120.0 / 110.0` 提为运行配置字段，使 FOV 与分辨率一同进入配置哈希，避免两者失配后无法追溯。

## 二、训练严重不足，GPU 算力大量闲置

### 实测

| 项 | 实测值 |
| --- | --- |
| 导出高斯数（PLY 头 `element vertex`） | **453,495** |
| 训练耗时 / 迭代数 | **735 秒** / 30,000 步 |
| 显存峰值 | **4,868 MiB**（RTX 4070 Ti SUPER 16 GiB） |
| `reconstruct` 耗时 | 13,071 秒 |
| 训练命令 | 仅覆盖 `--max-num-iterations`，其余全部默认 |

轨迹总长 18.48 单位（估算 50–90 米实景街段）。453k 高斯对这个尺度偏低约一个数量级，正常量级在 2–5 M。夜景 SfM 初始点云本身稀疏，而 Splatfacto 默认的梯度阈值稠密化在稀疏初始点上很难长出足够高斯，这与观测到的高斯数一致。显存只用掉 30%、训练时长只占整条流程 6%，说明 GPU 预算几乎没有被使用。

### 建议

```bash
ns-train splatfacto-big \
  --max-num-iterations 100000 \
  --pipeline.model.use-scale-regularization True \
  --pipeline.model.rasterize-mode antialiased \
  --pipeline.model.camera-optimizer.mode SO3xR3 \
  nerfstudio-data --data <dataset> --downscale-factor 1
```

| 改动 | 理由 |
| --- | --- |
| `splatfacto-big` | 更宽松的裁剪与稠密化阈值，直接提升高斯数 |
| `use_scale_regularization` | 抑制针状高斯，夜景漂浮噪点的主要形态之一 |
| `rasterize_mode antialiased` | 视图尺度跨度大（近处店面与远处街尾），减轻闪烁 |
| `camera-optimizer.mode SO3xR3` | 夜景 + 卷帘快门 + 防抖必然留下位姿残差，允许训练微调 |
| `--downscale-factor 1` | `nerfstudio-data` 在长边超过 1600 时自动降采样；改到 2048² 后不显式指定会吃掉第一节一半收益 |

以下两项收益可能更大，但**需先用 `ns-train splatfacto --help` 确认 nerfstudio 1.1.5 是否暴露；本次审核未能进入 WSL `3dgs` 环境核实**（底层 gsplat 1.4.0 具备相应能力）：

- `--pipeline.model.strategy mcmc --pipeline.model.max-gs 3000000`：MCMC 按给定预算填充高斯、不依赖梯度阈值，对"初始点云稀疏"这一具体失效模式尤其对症。
- `--pipeline.model.use-bilateral-grid True`：逐图外观补偿。本段为夜间行走，自动曝光与白平衡持续变化，可显著减少块状色斑与亮度呼吸。

### 内存约束

2152 张 2117² 图的训练缓存约 29 GB。主机物理内存 64 GiB 足够，但 **WSL2 默认只分配约一半（32 GB）**，余量过小。需要在 `.wslconfig` 中提高 `memory`，或采用第五节的减帧方案。

## 三、COLMAP 两处可以白拿的精度

### 3.1 精确已知的内参被 BA 改动

`transforms.json` 中 `fl_x = 276.9756`、`fl_y = 277.1690`。这两个值按构造**必须完全相同**——透视图是由全景合成的理想针孔投影，`pinhole_camera_parameters` 也是用同一个焦距同时写入 fx 与 fy。0.07% 的差异说明 mapper 把匹配噪声吸收进了焦距。在前向运动序列上，这是场景弯曲与尺度漂移的典型来源。

`cx`/`cy` 均为精确的 480.0，与 COLMAP `ba_refine_principal_point` 默认关闭一致，无需处理。

建议在 mapper 阶段固定全部内参：

```
--Mapper.ba_refine_focal_length 0
--Mapper.ba_refine_principal_point 0
--Mapper.ba_refine_extra_params 0
```

### 3.2 同一全景的 8 个共心视图被当作 8 个独立相机

同一全景派生的 8 个视图按构造共享同一光心。实测重建结果中它们的光心散布：

| 项 | 实测值 | 相对帧间基线 |
| --- | ---: | ---: |
| 中位散布 | 0.00182 单位 | 2.7% |
| p95 散布 | 0.00669 单位 | 9.8% |
| 最大散布 | 0.00832 单位 | 12.2% |
| 帧间基线（参照） | 0.0684 单位 | — |

由于真值为零，这一散布就是**位姿噪声下界，且达到工作基线的 2.7%（最坏 12%）**。三角化时该误差被 `1/tan(三角化角)` 放大，是几何发糊的直接来源。

COLMAP 3.8 自带 `colmap rig_bundle_adjuster`。把每个全景的 8 个视图声明为一个刚体 rig（相对位姿是纯旋转、零平移，完全已知）后，8 个自由位姿塌缩为 1 个位姿加固定 rig，该噪声可直接消除。这是本节收益最大、也是整份清单中改动最大的一项。

### 3.3 已核查、无需改动的项

记录下来以免后续朝错误方向调参：

- **sequential matching 的匹配跨度没有问题。** COLMAP 的 `SequentialMatching.quadratic_overlap` 默认开启，`overlap=16` 实际产生 1, 2, 4, 8, …, 2¹⁵ 的偏移阶梯。而 8 视图/帧的文件命名顺序恰好使偏移 8 对应"同一方向的下一帧"、16 对应隔两帧，一路覆盖到隔 32 帧，长基线图对是存在的。唯一浪费是每张图约 2/16 个落在同一全景内部的零基线图对（约 13%），不是主要矛盾。
- **本段不需要回环检测。** 检查了"时间上相隔 ≥20 帧、空间上相隔 <1 米"的帧对，结果为 **0 对**：这是一条弯曲单程路线，没有空间回访。轨迹水平包围盒 9.44 × 4.63 单位，路径长 18.48 单位，首尾直线距离 9.25 单位。更长的街段场景应开启 `loop_detection`，本段开启无收益。
- **全局几何正常。** 重力方向逐帧倾斜中位 0.86°；相机高度沿重力方向仅变化 0.173 单位（约 0.5–0.9 米），呈缓慢单调下降，符合步行起伏与轻微下坡，无显著漂移。

## 四、3.6 小时的 CPU COLMAP 是纯预算浪费

`reconstruct` 耗时 13,071 秒（3.63 小时），`train` 仅 879 秒。整条流程约 95% 的时间里 RTX 4070 Ti SUPER 处于空转，因为 `build_colmap_commands` 显式设置了 `--SiftExtraction.use_gpu 0` 与 `--SiftMatching.use_gpu 0`。

改用带 CUDA 的 COLMAP 构建（conda-forge 有 `colmap=*=*cuda*` 变体），特征提取与匹配通常快一个数量级。**这本身不提升质量，但释放出的时间预算正好支付第一节的 4.9× 像素量。** 这是本清单中唯一涉及环境变动的一项，需要修改 `environment.yml` 并重新生成 `conda-lock.yml`（按 README 的既有流程：conda-lock 4.0.2，先运行 `scripts/patch-conda-lock-manylinux.py`）。

若决定留在 CPU：`--SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1`（仿射 SIFT）对夜景弱纹理的匹配质量提升明显，但只能 CPU 运行且更慢，与 GPU 路线互斥，二选一。另外 `SiftExtraction.max_num_features` 默认 8192，夜景可提到 16384。

## 五、用帧数换分辨率

270 帧 / 90 秒，帧间基线 0.0684 单位（估算 0.25–0.35 米）。对 3 fps 步行采集而言该密度是冗余的；把帧数减半、分辨率提上去是更好的分配：

| 方案 | 图数 | 单图像素 | 训练缓存（估） | 中心角分辨率 |
| --- | ---: | ---: | ---: | ---: |
| 当前 | 2160 | 0.92 MP | 6 GB | 4.84 px/° |
| 270 帧 @2048² | 2160 | 4.19 MP | 27 GB | 10.3 px/° |
| **135 帧 @2048²** | **1080** | **4.19 MP** | **14 GB** | **10.3 px/°** |

135 帧方案使 COLMAP 图对数降至约 1/4、内存压力减半、帧间基线翻倍，同时保留全部分辨率收益。建议以此为第一个对比配置。

配套小改动：`media.py` 的 `create_blur_aware_subset`（每时间桶取最清晰帧）目前只在降级路径使用。既然要减帧，主路径直接改用它挑帧，可顺带避开运动模糊最重的帧。

## 六、素材层面的硬上限

以上全部完成后仍存在无法通过后处理突破的上限。依据 `capture-009-4k.yaml`：

- **相机未使用高分辨率档。** 记录为 Insta360 X6、双 1920×1920 镜头流、3840×1920 拼接、约 62 Mbps。清单备注"5.7K 与 8K 都是无新增细节的上采样"对**这个已录制文件**是正确的，但它同时说明录制时相机就设在了低分辨率模式。提高录制档位是唯一能突破 10.67 px/° 这一上限的手段。
- **夜景。** `luma_mean_median = 64.99`（0–255），高 ISO 噪声叠加步行长曝光。改在黄昏或白天重拍，对 SIFT 与 3DGS 都是量级差别。
- **单次前向直走。** 每个立面只被一条窄轨迹观察，视角张角很小。这是"贴着原路径看还行、视点稍微偏离就散架"的结构性原因，非训练参数可解。重点院落应采用两遍不同距离的通过加一圈慢速绕行，其收益远大于把单次通过的帧率加密。
- **行进速度。** 走慢可直接降低运动模糊，并提高有效帧密度。

这些属于下一次采集的输入要求，不构成对本次运行的返工项。README 已声明当前夜景素材不是正式质量标杆，本节与该声明一致。

## 建议实施顺序

按收益与改动量之比排序：

| 优先级 | 改动 | 涉及文件 | 是否变更环境 |
| ---: | --- | --- | :---: |
| 1 | 投影分辨率显式化，FOV 进配置 | `reconstruction.py`、`models.py` | 否 |
| 2 | 训练配置（`splatfacto-big`、正则、抗锯齿、相机优化、迭代数、显式 downscale） | `pipeline.py`、`models.py` | 否 |
| 3 | 固定内参三个 `ba_refine_*` 开关 | `reconstruction.py` | 否 |
| 4 | 帧数 270 → 135，主路径改用清晰度挑帧 | `models.py`、`pipeline.py` | 否 |
| 5 | `rig_bundle_adjuster` 刚体 rig 约束 | `reconstruction.py`（需新增 rig config 生成） | 否 |
| 6 | CUDA COLMAP | `environment.yml`、`conda-lock.yml` | **是** |

第 1–4 项属同一批改动，一次新 RunId 即可验证。第 5、6 项建议单独确认后再做，不与前四项混在同一批：第 5 项需要额外生成 rig 配置，第 6 项需要重做依赖锁。

按现有设计，任何配置变更都会改变配置哈希并强制新建运行，历史产物不会被覆盖，因此新旧配置可以并排比较。改动落地后需同步更新：

- `docs/RUNBOOK-009-MASKED.md` 中的运行配置块
- `README.md` 的阶段行为与耗时预估
- `tests/test_reconstruction.py`、`tests/test_pipeline.py` 中与投影分辨率、COLMAP 参数、训练命令相关的断言

## 未核实项

以下判断需要在 WSL `3dgs` 环境中确认后才能作为实施依据：

- nerfstudio 1.1.5 的 `splatfacto` 是否暴露 `strategy`（`mcmc`）、`max_gs` 与 `use_bilateral_grid`（第二节）。
- `splatfacto-big` 在本版本中的具体参数覆盖范围。
- `FullImageDatamanager` 的 `cache_images` 默认值，直接决定第二节的内存估算是落在显存还是主机内存。
- 提高投影分辨率后 COLMAP `SiftExtraction.max_image_size`（默认 3200）是否仍不触发内部降采样：2048 与 2117 均在限内，但需在实际日志中确认。
