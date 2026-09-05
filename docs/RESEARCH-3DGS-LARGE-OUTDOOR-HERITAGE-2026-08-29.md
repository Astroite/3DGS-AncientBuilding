# 大范围室外古建筑 3DGS：互联网检索原始记录

- 检索日期：2026-08-29（Asia/Shanghai）
- 面向项目：GSDB，Insta360 360° 视频 → SfM/3DGS → 审核与发布
- 结果性质：公开网页、论文页面、作者代码仓库、官方文档与行业指南的结构化检索记录
- 配套规范：[STANDARD-3DGS-LARGE-OUTDOOR-HERITAGE.md](STANDARD-3DGS-LARGE-OUTDOOR-HERITAGE.md)

> “原始结果”在本文中指检索式、来源元数据、来源原意的短摘要、适用性和限制，不复制受版权保护的长段正文。网页和仓库会更新；任何版本、许可和兼容性结论在正式采用前都应再次核验。

## 1. 检索范围与证据分级

本次检索同时覆盖：

1. 360°/全景图像的大范围室外 3DGS；
2. 常规相机、无人机和分块式大场景 3DGS；
3. 古建筑、建筑遗产和文化遗产的 3DGS 案例；
4. 摄影测量、坐标、尺度、控制点和数字遗产质量指南；
5. 采集、SfM、训练、评测、压缩、LOD 和 Web/引擎发布；
6. 可供 Agent/Codex 使用的现成 3DGS skills。

证据分级：

- **A**：标准组织/官方产品文档、作者官方仓库、正式同行评审论文或权威遗产机构指南；可作为规范性依据。
- **B**：作者项目页、预印本或新近研究实现；适合试验与设计参考，不能自动升级为生产标准。
- **C**：厂商教程、社区仓库或非官方实现；适合作为操作线索，关键结论必须回到 A/B 来源复核。

## 2. 代表性检索式

以下是本轮使用和扩展过的检索主题；同一主题还以 `GitHub`、`paper`、`official`、`heritage`、`panorama`、`outdoor`、`large-scale` 等限定词交叉搜索：

```text
large scale outdoor 3D Gaussian Splatting workflow
panoramic 360 outdoor Gaussian Splatting large scene
equirectangular panorama 3DGS partitioning
ancient architecture cultural heritage 3D Gaussian Splatting
historical building UAV photogrammetry 3DGS
cultural heritage 3DGS LiDAR accuracy workflow
large scale Gaussian Splatting block partition official code
hierarchical Gaussian Splatting large datasets official
CityGaussian official repository
distributed Gaussian Splatting large scene OOM
3DGS geometry evaluation LiDAR point cloud
COLMAP panorama rig official documentation
Nerfstudio 360 video custom data
gsplat memory evaluation official
3DGS compression PLY SPZ SOG glTF standard
KHR_gaussian_splatting specification
3DGS skill SKILL.md GitHub
Gaussian Splatting engineering guide agent skill
heritage photogrammetry good practice control points overlap
ICOMOS 3D digitisation quality cultural heritage
```

没有发现一个被行业普遍采纳、能从采集一直覆盖到归档的“3DGS 国际标准工作流”。当前可依赖的是：摄影测量/数字遗产规范、方法作者的公开流程，以及逐步成熟的高斯交换和流式格式；本文配套规范是这些来源与本项目实测的综合，而不是冒充外部标准。

## 3. 最相关结果：360° 全景大范围室外

### R01 — PanoLOG / G²PS

- URL：[项目页](https://insta360-research-team.github.io/GGPS-Website/) · [代码](https://github.com/Insta360-Research-Team/GGPS) · [arXiv](https://arxiv.org/abs/2607.08769)
- 时间/机构：2026，Insta360 Research 等；证据级别 B（预印本 + 作者代码）。
- 原始结果摘要：等距柱状投影（ERP）相机几乎“看见所有方向”，导致依赖局部相机视锥的大场景分区退化成近似全局训练。方法采用全局粗模与分块精修两阶段：粗阶段引入天空球和全景单目深度，精修阶段按视差/几何不确定性生成自适应边界，并按梯度重要性给分块分配相机。
- 公开流程：准备 openMVG/COLMAP 数据；可选生成 DAP 深度与天空 mask；对单目深度做尺度对齐；全局 coarse 训练；分区；块内并行微调；合并；渲染；计算指标。仓库示例按每 8 帧留 1 帧测试。
- 对本项目的价值：这是检索到的、与“360 视频 + 大范围室外”最直接匹配的方法，尤其说明不能简单把 CityGaussian 的视锥分块原样套到全景输入上。
- 限制：参考环境为 Python 3.9、PyTorch 2.8/CUDA 12.8、RTX 5090 D，与本项目当前 Nerfstudio/CUDA 11.8 环境不兼容；应隔离做 PoC。项目页文字内容是 CC BY 4.0，但算法、权重和相关知识产权另有保留；代码仓库许可必须以仓库中的最新 LICENSE 为准。

### R02 — COLMAP 原生 Panorama/Rig 工作流

- URL：[Rig Support](https://colmap.github.io/rigs.html) · [Tutorial](https://colmap.github.io/tutorial) · [Output Format](https://colmap.github.io/format.html)
- 证据级别：A（COLMAP 官方文档）。
- 原始结果摘要：新版 COLMAP 支持 rig/frame 建模，并给出从 360° 全景重建的 `python/examples/panorama_sfm.py` / `pycolmap.panorama.reconstruct` 工作流；全景被映射成一个虚拟透视相机 rig。视频可用 sequential matching，图像名需保持顺序；大型集合可引入词汇树与回环匹配。
- 对本项目的价值：验证“同一全景拆成多个虚拟 PINHOLE，相对外参固定”的建模方向，也提供从 COLMAP 3.8 自维护 rig 逻辑迁移到新版原生 rig 的 A/B 路线。
- 限制：新版本行为、数据库和模型格式需在隔离分支验证，不能无测试替换当前已固定的 COLMAP 3.8 基线。

### R03 — Nerfstudio 自定义 360 数据指南

- URL：[Custom Dataset 文档](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/custom_dataset.md)
- 证据级别：A（官方仓库文档）。
- 原始结果摘要：360 图像/视频可展开为多张透视图；默认 8 个视图，细节或配准不足时可尝试 14 个；视频目标帧数约为视频秒数的 3 倍；建议裁掉底部约 20%，并将相机举过头顶采集。
- 对本项目的价值：与当前 3 fps、8 视图、14 视图降级路径和 20% 底裁高度一致，说明现有输入预处理基线不是主要偏差来源。
- 限制：这是通用快速入门，不是古建筑测量规范，也没有解决长距离全景的分区退化。

## 4. 大场景训练与分块方法

### R04 — VastGaussian

- URL：[项目页](https://vastgaussian.github.io/)
- 时间：CVPR 2024；证据级别 A/B（正式论文 + 作者项目页）。
- 原始结果摘要：将大场景渐进分成多个 cell；以“空域感知的可见性”为分区分配训练相机；独立优化后合并；以解耦外观建模降低不同图像曝光差异。
- 适用性：常规透视相机的大型街区/园区分块思路清晰。
- 限制：项目页未给出可核验的作者官方训练代码链接；更重要的是，全景相机的全向可见性会削弱按局部视锥分配相机的前提。

### R05 — Hierarchical 3D Gaussians（H3DGS）

- URL：[作者官方仓库](https://github.com/graphdeco-inria/hierarchical-3d-gaussians)
- 时间：SIGGRAPH 2024；证据级别 A。
- 原始结果摘要：先完成全局 COLMAP，再切 chunk、做块内 BA、生成单目深度和 mask；训练全局粗 scaffold、并行训练各块、构建与优化层级，最后合并。查看器可按显存预算加载 LOD。
- 适用性：超大数据集、多尺度漫游和单机查看；层级表达比单一巨型 PLY 更适合园区/古镇浏览。
- 限制：流水线复杂，依赖外部深度/分块和专用查看器；采用前需复核许可、GPU 和数据格式。

### R06 — CityGaussian / CityGaussianV2

- URL：[作者官方仓库](https://github.com/Linketic/CityGaussian)
- 时间：ECCV 2024、ICLR 2025；证据级别 A。
- 原始结果摘要：面向城市尺度场景的分区训练、数据分配和多 GPU 训练；V2 增加联合位姿优化、2DGS 几何、Mip-Splatting/AbsGS 等组件。仓库 FAQ 提示块内相机太少会使边界估计失真，并给出下采样、缓存限制和剪枝等 OOM 处理方向。
- 适用性：无人机/透视相机的大街区，或需要几何评估与位姿联合优化的场景。
- 限制：仓库为 CC BY-NC-SA 4.0；360 数据不可直接照搬其视锥分块。

### R07 — DOGS

- URL：[作者官方仓库](https://github.com/AIBluefisher/DOGS)
- 时间：NeurIPS 2024；证据级别 A。
- 原始结果摘要：以分布式高斯共识支持多 GPU 大场景训练。
- 适用性：计算集群可用且单模型无法容纳时。
- 限制：会显著增加调度、同步和复现实验成本；不是本项目单张 16 GB GPU 的首选路径。

### R08 — Grendel-GS

- URL：[作者官方仓库](https://github.com/nyu-systems/Grendel-GS)
- 时间：ICLR 2025 Oral；证据级别 A。
- 原始结果摘要：分布式多 GPU 训练，并支持 batch size 大于 1 的扩展训练。
- 适用性：拥有稳定多 GPU 基础设施时做规模扩展。
- 限制：解决的是计算扩展，不会自动修复数据质量、天空、错位或错误分区。

### R09 — GS-Scale

- URL：[作者官方仓库](https://github.com/SNU-ARC/GS-Scale)
- 时间：ASPLOS 2026；证据级别 A。
- 原始结果摘要：将部分状态卸载到主机内存，以较低 GPU 显存训练更大的高斯集合；作者报告显存降低并面向消费级 GPU。
- 适用性：单卡因高斯状态 OOM、主机内存充足，且接受更慢数据搬运时。
- 限制：仓库当前支持条件偏窄（CUDA 12.x，且说明了 CPU 平台约束）；小场景仍建议纯 GPU。先用小场景证明兼容和数值一致性。

### R10 — CLM-GS

- URL：[作者官方仓库](https://github.com/nyu-systems/CLM-GS)
- 时间：ASPLOS 2026；证据级别 A。
- 原始结果摘要：CPU/GPU 协同和内存卸载，目标是突破单 GPU 可容纳的高斯数量。
- 适用性：真正达到数千万至上亿高斯、显存成为确定性瓶颈时。
- 限制：工程复杂度和主机内存/带宽要求高；不应在当前模型异常收缩或评测失真的情况下先引入。

### R11 — A LoD of Gaussians

- URL：[作者项目页](https://felixwindisch.github.io/ALoDOfGaussians/)
- 时间：SIGGRAPH 2026；证据级别 A/B。
- 原始结果摘要：面向超大场景的外存与 LOD 路线，强调不依赖传统空间分块的单 GPU 浏览/处理能力。
- 适用性：超大模型的多尺度查看和外存管理设计参考。
- 限制：较新，生态和可复现性需继续观察。

### R12 — NVIDIA HiGS

- URL：[NVIDIA Research 项目页](https://research.nvidia.com/labs/sil/projects/higs/)
- 时间：2026；证据级别 A/B。
- 原始结果摘要：层级式高斯渲染加速，在大模型上保持精确合成，重点是渲染而非训练。
- 适用性：模型已经正确、瓶颈位于实时浏览时。
- 限制：不能用它解决 SfM、训练稳定性、几何或尺度问题。

## 5. 训练器、外观、抗混叠与几何

### R13 — Nerfstudio Splatfacto

- URL：[Splatfacto 文档](https://docs.nerf.studio/nerfology/methods/splat.html) · [实现源码](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py)
- 证据级别：A。
- 原始结果摘要：以 SfM 点初始化通常优于随机初始化；官方文档给出 `splatfacto` 与更高质量、更多显存的 `splatfacto-big` 基线；实现包含相机优化、双边网格、不同 densification 策略和抗混叠选项。
- 对本项目的价值：继续保留为当前可靠基线；双边网格适合单次漫游中的自动曝光/白平衡变化。
- 限制：通用方法不包含 360 大场景专用分区；单纯增加训练步数不是规模化方案。

### R14 — Splatfacto-W

- URL：[官方文档](https://docs.nerf.studio/nerfology/methods/splatw.html)
- 证据级别：A。
- 原始结果摘要：面向非受控照片集合的外观变化和瞬态干扰。
- 适用性：跨天、跨季节、多相机、游客和光照差异明显的照片集合。
- 限制：同一次 360 视频漫游优先使用采集约束、mask 和较轻的曝光建模，不必默认承担 W 模型的额外复杂度。

### R15 — gsplat

- URL：[官方文档](https://docs.gsplat.studio/) · [评测页](https://docs.gsplat.studio/main/tests/eval.html) · [仓库](https://github.com/nerfstudio-project/gsplat)
- 证据级别：A。
- 原始结果摘要：提供 CUDA rasterization、稀疏梯度、分布式训练、抗混叠和 MCMC 等选项；官方评测同时报告 PSNR/SSIM/LPIPS、训练时间和峰值内存。
- 适用性：可作为降低显存、提升吞吐和标准化性能评测的基础库。
- 限制：官方 benchmark 的数据集、硬件和配置不能直接外推到本项目；版本升级必须用同一场景、同一留出集做 A/B。

### R16 — Mip-Splatting

- URL：[作者官方仓库](https://github.com/autonomousvision/mip-splatting)
- 时间：CVPR 2024；证据级别 A。
- 原始结果摘要：通过 3D 平滑与 2D mip filter 降低采样频率变化引起的锯齿和尺度伪影。
- 适用性：古建密集瓦片、栏杆、窗棂和远近尺度变化大的浏览。
- 限制：抗混叠不能修正原始位姿或重建几何错误。

### R17 — WildGaussians

- URL：[作者项目页](https://wild-gaussians.github.io/)
- 时间：NeurIPS 2024；证据级别 A/B。
- 原始结果摘要：用鲁棒特征与图像/高斯外观模型处理遮挡和照片间外观变化。
- 适用性：多时段、多天气、多人流的互联网照片或长期采集。
- 限制：会把真实材质、光照差异和模型自由度纠缠；文物颜色记录必须保留原始影像和颜色管理信息。

### R18 — 2D Gaussian Splatting

- URL：[作者官方仓库](https://github.com/hbb1/2d-gaussian-splatting)
- 时间：SIGGRAPH 2024；证据级别 A。
- 原始结果摘要：以 2D surfel 提升表面几何一致性，并提供无界场景 mesh 提取方向。
- 适用性：需要几何/网格派生物，而不只是视觉漫游时，作为独立对照分支。
- 限制：不能因其“几何更好”就取代测量控制、LiDAR/TLS 或精度验证。

### R19 — 原始 3DGS

- URL：[作者官方仓库](https://github.com/graphdeco-inria/gaussian-splatting)
- 时间：SIGGRAPH 2023，仓库持续维护；证据级别 A。
- 原始结果摘要：3DGS 基础实现；后续仓库更新加入曝光补偿及抗混叠相关能力。
- 适用性：定义术语、核对默认策略和做复现基准。

## 6. 文化遗产与古建筑案例

### R20 — 3DGS 与 LiDAR 的建筑遗产集成工作流

- URL：[TU Delft 论文页](https://research.tudelft.nl/en/publications/from-comparison-to-integration-a-workflow-evaluation-of-3d-gaussi/)
- 时间：Automation in Construction，2025；证据级别 A。
- 原始结果摘要：3DGS 在视觉表现和交互响应方面有优势，LiDAR 点云在结构精度和分割方面更可靠；提出将两者集成，而非把 3DGS 视为 LiDAR 的替代品。
- 规范含义：文化遗产项目应将 3DGS 定位为可视化/传播层；若目标包括测绘、病害判断或修缮依据，必须保留独立的度量几何层和验证链。

### R21 — 无人机摄影测量 + 3DGS 的历史建筑再现

- URL：[npj Heritage Science](https://www.nature.com/articles/s40494-026-02649-7)
- 时间：2026；证据级别 A。
- 原始结果摘要：面向东亚历史建筑的 UAV 摄影测量和 3DGS/UE5 可视化，讨论遮挡、弱纹理和琉璃瓦等非朗伯材质；流程包含云端到本地、浮点伪影清理和高斯属性优化。
- 规范含义：檐下、屋面、院落角点需专门补拍；反光瓦面、天空和植被必须作为独立 QA 类别。论文中的相对一致性不能直接等价为测绘级绝对精度。

### R22 — 高保真文化遗产重建的图像增强工作流

- URL：[Heritage Science DOI](https://doi.org/10.1038/s40494-026-02355-4)
- 时间：2026；证据级别 A。
- 原始结果摘要：低质量图像先破坏 SfM，再经位姿漂移和噪声深度影响 3DGS；论文组合了图像增强、Dense-SfM、体素/高斯混合和渐进训练。
- 规范含义：先解决输入与位姿，不要只在 3DGS 末端调参。
- 限制：生成式超分可能虚构文物纹理，不应成为档案主流程；如试验，必须保留原片、标记衍生关系，并禁止把增强图当作原始证据。

### R23 — MGSO：古建场景的位姿与高斯联合优化

- URL：[Heritage Science DOI](https://doi.org/10.1038/s40494-026-02675-5)
- 时间：2026；证据级别 A。
- 原始结果摘要：针对大范围室外遗产中的光照不均、镜面反射和弱纹理，以 view-aware 约束、点图位姿初始化和联合位姿/高斯优化提高稳健性；同时评估轨迹和新视角指标。
- 规范含义：位姿质量与 3DGS 质量应联合观察；只看渲染 PSNR 不足以证明空间正确。

### R24 — Patan Durbar Square 3DGS 案例

- URL：[ScienceDirect 论文页](https://www.sciencedirect.com/science/article/abs/pii/S2212054825000943)
- 时间：2026；证据级别 A（论文摘要页）。
- 原始结果摘要：以手机短视频采集文化遗产地点，COLMAP 后训练多种 Gaussian Splatting，并比较伪影清理/分割等流程。
- 规范含义：消费级采集可用于可视化，但短视频成功案例不能替代大型古镇的分段、尺度和控制要求。

### R25 — CULTURE3D 数据集

- URL：[ICCV 2025 论文 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_CULTURE3D_A_Large-Scale_and_Diverse_Dataset_of_Cultural_Landmarks_and_ICCV_2025_paper.pdf)
- 时间：ICCV 2025；证据级别 A。
- 原始结果摘要：覆盖多类文化地标并组织从图像采集、匹配/SfM、稠密几何、3DGS baseline 到地图对齐、BA 和评测的链条。
- 适用性：作为文化地标算法的外部验证集/流程参考，避免只在自有场景上优化。
- 限制：数据分布、许可、尺度基准和本项目 360 输入差异需另行核验。

### R26 — GauU-Scene：无人机 + LiDAR 大场景基准

- URL：[arXiv](https://arxiv.org/abs/2401.14032)
- 时间：2024；证据级别 B。
- 原始结果摘要：覆盖超过 1.5 km² 的无人机 RGB 与 LiDAR；通过粗配准后 ICP 对齐 SfM 与 LiDAR；LiDAR 先验对几何的改善比训练视图图像指标更明显。
- 规范含义：训练图像 PSNR 可能掩盖几何问题；具有测绘目标时必须对点云/控制进行独立几何评测。

### R27 — 3DGS Survey / Drum Tower 自定义场景

- URL：[项目页](https://autumn119.github.io/3DGS-Survey/)
- 证据级别 B。
- 原始结果摘要：以 PSNR、SSIM、LPIPS、FPS、VRAM、训练时间等多维指标比较 3DGS 方法，并包含大型室外/鼓楼类型场景的讨论。
- 适用性：建立多指标实验表格的参考。
- 限制：综述网页不是遗产测量标准，具体数值需回到论文和代码复核。

## 7. 文化遗产摄影测量与 3D 质量指南

### R28 — Historic England 摄影测量良好实践

- URL：[Photogrammetric Applications for Cultural Heritage](https://historicengland.org.uk/images-books/publications/photogrammetric-applications-for-cultural-heritage/)
- 证据级别 A（官方遗产机构指南）。
- 原始结果摘要：从项目目标、摄影网络、相机/镜头、控制、处理到交付说明文化遗产摄影测量的良好实践。
- 规范含义：采集设计必须由成果用途和精度倒推；控制点、冗余、多角度覆盖和记录链不能被“视频更方便”取代。

### R29 — Historic England 地理空间测量规范

- URL：[Geospatial Survey Specifications for Cultural Heritage](https://historicengland.org.uk/images-books/publications/geospatial-survey-specifications-cultural-heritage/heag317-geospatial-survey-specifications-cultural-heritage/)
- 证据级别 A。
- 原始结果摘要：包含影像重叠、颜色/RAW、控制点和不同精度等级的要求；这些数值随任务等级而变化。
- 规范含义：本规范采用“目标精度必须在项目开始前声明并以检查点验证”，不把某一个示例毫米阈值误写成所有古建项目的通用要求。

### R30 — ICOMOS 3D 数字化质量研究

- URL：[完整报告 PDF](https://www.icomos.org/images/DOCUMENTS/Study_on_3D_Quality_Digitisation_Final_Study_Report.pdf) · [发布说明](https://www.icomos.org/news/new-study-on-quality-in-3d-digitisation-of-tangible-cultural-heritage/)
- 证据级别 A（ICOMOS）。
- 原始结果摘要：3D 数字化质量取决于目的、对象复杂度、视点/设备冗余、元数据、处理链、验证和长期保存，而不只是模型的视觉逼真度。
- 规范含义：必须保存原始资料、转换关系、版本、坐标、单位、方法和验收记录；展示模型不能成为唯一档案。

## 8. 采集与发布格式

### R31 — Postshot 采集指南

- URL：[Capturing Guidelines](https://activation.jawset.com/docs/d/Postshot%2BUser%2BGuide/Capturing%2BGuidelines)
- 证据级别 C（商业软件操作指南）。
- 原始结果摘要：要求静态场景、丰富视角、足够重叠、避免运动模糊、失焦、耀斑、闪光和移动阴影；坏图可能比少图更糟；视频抽帧建议落在每秒数帧的量级。
- 规范含义：可用于现场速查，但文化遗产正式采集仍以 R28–R30 为上位依据。

### R32 — glTF `KHR_gaussian_splatting`

- URL：[Khronos 新闻稿](https://www.khronos.org/news/press/gltf-gaussian-splatting-press-release) · [扩展规范](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md)
- 时间：2026-02-03 发布候选；证据级别 A。
- 原始结果摘要：Khronos 发布 glTF 高斯扩展候选，以标准生态表达和传输高斯场景。
- 规范含义：适合做发布衍生格式，但在正式批准和工具链稳定前，不宜作为唯一长期保存格式。

### R33 — SPZ

- URL：[Niantic Labs 官方仓库](https://github.com/nianticlabs/spz)
- 证据级别 A。
- 原始结果摘要：3DGS 的压缩格式/库，面向较小体积；需要严格处理坐标轴、旋转和球谐约定。
- 规范含义：用于交付副本，转换前后必须跑坐标、外观和数量 QA。

### R34 — SOG / Streamed SOG

- URL：[PlayCanvas 格式指南](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/) · [Streamed SOG](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/streamed-sog/)
- 证据级别 C（厂商官方文档）。
- 原始结果摘要：将 PLY 作为源/编辑输入，SOG 作为压缩运行时格式；Streamed SOG 将非常大的场景切成空间块并支持 LOD 流式加载。
- 规范含义：适合 Web 发布和大模型浏览；不可只归档有损运行时文件。

## 9. 可复用 Agent Skills 检索结果

### R35 — Awesome-Gaussian-Skills

- URL：[仓库](https://github.com/jaccen/Awesome-Gaussian-Skills) · [3dgs-engineering-guide](https://github.com/jaccen/Awesome-Gaussian-Skills/blob/main/skills/3dgs-engineering-guide/SKILL.md) · [3dgs-training-debugger](https://github.com/jaccen/Awesome-Gaussian-Skills/blob/main/skills/3dgs-training-debugger/SKILL.md) · [3dgs-experiment-planner](https://github.com/jaccen/Awesome-Gaussian-Skills/tree/main/skills/3dgs-experiment-planner)
- 时间/许可：2026 年仍在更新；仓库与所检视 skills 标注 Apache-2.0；证据级别 C（社区知识包）。
- 原始结果摘要：仓库自述收录 700+ 方法和约 12 个 skills。工程指南可按 `culture`、`pipeline`、`best-practices`、`pitfalls`、`decision-tree` 等维度加载；另有训练调试、实验规划、方法比较、代码审查和部署类 skills。
- 优点：作为方法索引和 Agent 路由器很实用；`3dgs-engineering-guide` 明确要求加载引用片段、禁止编造指标，并覆盖文化遗产分类。
- 风险：不是本项目从 360 视频到遗产 QA 的端到端 skill；训练调试器包含大量“健康轨迹”和硬阈值，这些值高度依赖数据、实现和硬件，不能未经 primary source 或本项目基线验证就写入门禁。
- 采用建议：只读审计后选择性借鉴；若安装，固定 commit、记录许可证与文件 hash，不执行未知安装脚本，更不使用 `curl | bash`。本轮没有安装或执行第三方 skill。

### R36 — SparkJS Skill

- URL：[SKILL.md](https://github.com/shi3z/sparkjs-skill/blob/master/SKILL.md)
- 证据级别 C。
- 原始结果摘要：服务于 SparkJS/Web 端高斯场景渲染和格式使用。
- 适用性：Web 查看器和 LOD/格式集成。
- 限制：不是采集、SfM、训练或文化遗产质量 skill。

### R37 — KIRI Maker

- URL：[仓库](https://github.com/willjim/KIRI-Maker)
- 证据级别 C。
- 原始结果摘要：围绕高斯查看、相机路径和内容制作的工具/skill。
- 适用性：预览和镜头制作。
- 限制：不覆盖大范围 360 重建主流程。

### R38 — OpenAI Skills API

- URL：[OpenAI 官方 Skills API 参考](https://developers.openai.com/api/reference/go/resources/skills)
- 证据级别 A。
- 原始结果摘要：Skill 是可创建、列出、读取并具有版本的资源包；这为把本项目标准封装为可审计、可版本化的项目 skill 提供了产品层依据。
- 结论：公开检索中没有找到完全匹配“360 古建长视频 → 画质筛选 → 人/天空 mask → panorama rig SfM → 全景专用大场景分区 → 训练 → 遗产双轨 QA → 多格式发布”的 skill。最合适的后续动作是基于本文规范创建本项目专用 skill，而不是直接把通用调参 skill 当作标准。

## 10. 与当前 GSDB 实跑的交叉核对

本节不是互联网来源，而是为了判断外部方法是否真正解决本项目问题所做的本地事实核对。

当前 `yunxiu-20260822 / vid-20260823-104518-00-014` 运行记录显示：

- 427 秒输入，641 张全景，5128 张透视视图；注册率 99.14%，单一连通分量；
- 训练约 12.85 小时，首次 OOM 后以 `downscale=2` 重试；峰值显存约 15.76 GiB，磁盘约 44.70 GiB；
- 最终导出仅 19,184 个高斯，低于项目警告线；
- `metric_scale=false`；朝向和发布高度跨度检查未通过，靠 unsafe override 才继续；
- 当前所谓 eval curve 是从 TensorBoard 中抓取所有 `eval*` tag 后按 step 罗列，尚未证明是固定图像集上的聚合曲线。按 5k step 窗口重算后指标显著波动，不支持“100k 已稳定收敛”的强结论。

因此，当前最主要风险不是“没有训练到足够步数”，而是：

1. 评测设计不能区分场景/相机差异与训练进展；
2. 超长全景轨迹还没有专用空间分块；
3. 天空、远景和单目深度先验缺失；
4. 没有度量尺度/控制点，且最终坐标安全检查失败；
5. OOM 降采样后高斯集合异常偏小，需先诊断 densification/culling/导出链，而不是继续扩大步数。

## 11. 检索结论（未经项目化改写前的原始判断）

1. **有直接相关分享**：PanoLOG/G²PS 是当前最贴合 360° 大范围室外的公开方案；H3DGS、CityGaussian、VastGaussian 是常规相机大场景的重要参照。
2. **有古建案例，但缺统一标准**：文化遗产论文普遍证明视觉传播价值，同时反复强调位姿、遮挡、反射、弱纹理和度量几何问题。3DGS 不应独自承担测绘/修缮证据角色。
3. **当前项目预处理基线基本合理**：Nerfstudio 官方 360 指南与 3 fps、8/14 透视视图和 20% 底裁接近；升级重点应转向评测、天空、分块和尺度。
4. **大场景扩展要区分相机类型**：全景相机不能默认使用基于局部视锥的分块；先做语义/空间段落切分，再试全景专用 G²PS。
5. **内存技术是第二层优化**：gsplat、GS-Scale、CLM-GS、DOGS/Grendel-GS 能扩展内存或多卡，但不会修复错误位姿、天空和错误验收。
6. **有相似 skills，没有完整匹配 skill**：Awesome-Gaussian-Skills 最接近知识库和调试助手，但仍需以主来源和本项目实测校验。推荐另建项目专用 skill。
7. **发布格式仍在成熟**：长期保存应保留全精度 PLY、相机/变换、配置、原始影像和 hash；SPZ/SOG/streamed SOG/glTF 扩展仅作为可再生成的发布副本。

## 12. 未作为规范性依据的结果

- Reddit、论坛和无来源转载：可用于发现关键词，但不作为门禁依据。
- 只有展示视频、无论文/代码/参数的方法：不纳入生产建议。
- 非作者的 VastGaussian 复现：未作为官方实现引用。
- 任意单一数据集上的“最佳 PSNR”“节省显存百分比”：没有同数据、同图像、同硬件 A/B 时不外推。
- 生成式超分/去模糊：可能改变文物表面证据，不进入档案主链。

