# BiomedCLIP：1027 方案与胸腺瘤逐步实验

本仓库保存《改进建议1027》对应的当前代码：多层密集定位头、正常/真实异常/动态伪异常混合取样、图像与特征伪异常、排序损失、患者划分审计和独立训练评估。

旧文档前三步的独立归档见 [first-edit](https://github.com/hiAhi233/first-edit)。本仓库保留其基础包作为新方案依赖，但当前实验入口是 `run_brain_novelty.py`，不要将两套方案混用。

## 主要文件

| 功能 | 文件 |
|---|---|
| 患者划分、支持集、元数据审计 | `prepare_patient_splits.py`、`paclip/data/patient_manifest.py` |
| 基准训练、新方案训练与预检 | `run_brain_novelty.py` |
| 三层密集定位头、冻结与微调策略 | `paclip/models/dense_localizer.py`、`paclip/models/novelty_model.py` |
| 固定来源配额与动态合成 | `paclip/data/mixed_sampling.py`、`synthetic_image.py`、`synthetic_feature.py` |
| 分来源定位损失与排序项 | `paclip/losses_novelty.py` |
| 训练、缓存、断点及基准身份 | `paclip/engine/novelty_trainer.py`、`novelty_cache.py`、`novelty_provenance.py`、`paclip/implementation_identity.py` |
| 定位、病灶召回、误报、FROC 与评估条件 | `paclip/evaluation/novelty_protocol.py`、`localization_budget.py`、`novelty_baseline.py` |
| 完整指标汇总 | `report_brain_novelty.py` |
| 本轮六组配置与顺序执行脚本 | `configs/doc1027_staged/`、`run_doc1027_stages.ps1` |

## 环境

本机验证环境：Windows、Python 3.10.20、RTX 5060 Laptop GPU，PyTorch `2.12.0.dev20260408+cu128`、torchvision `0.27.0.dev20260407+cu128`。这两个 nightly 版本是实际环境记录，不是其他机器必须使用的版本。

先安装适合设备且相互兼容的 `torch`、`torchvision`，再执行：

```powershell
python -m pip install -r requirements.txt
```

BiomedCLIP 主干为 `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`，模型和 tokenizer 需单独准备。入口默认设置 Hugging Face 离线模式，因此运行前应有完整本地缓存；需要联网加载时，可在启动前显式设置 `HF_HUB_OFFLINE=0`、`TRANSFORMERS_OFFLINE=0`。

仓库不包含医学影像、掩码、患者清单、训练权重、缓存或病例图。正式训练需要自己的数据和已经核实的元数据。

## 今天实际运行的设置

**请使用 `configs/doc1027_staged/`。** 这些配置从本轮控制脚本提取，合并程序默认值后，已与六组实际保存的 `config.json` 逐项核对一致。`configs/novelty/` 保留早期配置以兼容历史实验及回归测试，其中含冻结视觉适配器等旧设置，不能代表本轮实验。

共同设置：胸腺瘤 CT，seed=0，真实支持集总计 K=5（2 正常、3 异常），batch=8，每组训练2700步。各组独立从同一第一步基准初始化视觉适配器，新头使用相同随机初始化，不将上一组训练权重传给下一组。冻结原始主干与文本分支，训练新头及已有视觉适配器。

三层为0起始 block 5/8/11，各自512→128后拼接；使用 `doc1027` 解码器。每300步按源验证全图 Pixel AP 选择检查点。二值阈值从源验证集选择，测试集不参与选参。

| 对照含义 | 配置文件名 | 正常 / 真实异常 / 图像合成 / 特征合成 | 排序权重 |
|---|---|---|---:|
| 仅多层密集定位头 | `dense_only.json` | 4 / 4 / 0 / 0 | 0 |
| 无合成取样比例对照 | `ratio_control.json` | 6 / 2 / 0 / 0 | 0 |
| 特征空间伪异常 | `feature_synthetic.json` | 4 / 2 / 0 / 2 | 0 |
| 图像空间伪异常 | `image_synthetic.json` | 4 / 2 / 2 / 0 | 0 |
| 两类伪异常联合 | `joint_synthetic.json` | 4 / 2 / 1 / 1 | 0 |
| 联合再加排序损失 | `joint_rank.json` | 4 / 2 / 1 / 1 | 0.2 |

特征合成使用逐维高斯 σ=0.05/0.10/0.20，保留视觉梯度；图像合成包含亮、暗、模糊、局部噪声、纹理替换、膨胀、压缩和混合信号。排序模式为 `global_mean`，按正样本图像平均。具体参数以配置和核心实现为准。

## 数据准备与审计

输入采用 JSONL 或 JSON 清单，每条样本包含 `sample_id`、经核实的全局 `patient_id`、`image_path`、异常样本的 `mask_path`、`label`、`dataset`、`sequence`、`disease_labels` 和核验来源 `metadata_source`。正式清单要求真实的 `patient_verified`、`sequence_verified`、ROI 核验信息及 `train/validation/test` 划分。相对图像路径按清单所在目录解析。

病灶标签、患者身份、序列和 ROI 必须依据实际资料核实；不能只为通过审计而填写认证标志。没有单独 ROI 文件时使用灰度阈值策略，这不是解剖分割。无可信体积信息时只能做二维评价，不能将无序 PNG 堆叠后声称三维结果。

以下路径都是需要使用者准备或替换的示例，不是仓库内已有的数据：

```powershell
# 已核实并填写划分的清单：不传 --assign-splits。
python prepare_patient_splits.py `
  --manifest data/verified_metadata.jsonl --seed 0 --output metadata/patient_split

# 若需要按患者生成划分，加入 --assign-splits；不会自动反复换种子。
python run_brain_novelty.py preflight `
  --manifest metadata/patient_split/split_manifest.jsonl `
  --support metadata/patient_split/support_manifest.json `
  --output runs/preflight.json
```

`--scan-root` 只生成待核验清单，不能据此直接获得患者独立性认证。正式入口会检查患者交叉、图像重复、支持集来源和类别、ROI 可用性等条件。

## 运行顺序

### 1. 训练患者独立的第一步基准

```powershell
python run_brain_novelty.py baseline `
  --manifest metadata/patient_split/split_manifest.jsonl `
  --support metadata/patient_split/support_manifest.json `
  --task thymoma --organ thymoma --prompt-set thymoma_sentence `
  --seed 0 --steps 2700 --device cuda --output runs/doc1027/baseline
```

该入口从原始 BiomedCLIP 预训练权重建立基准，保存 `baseline_provenance.json` 及统一评估结果。新头训练会校验基准步数、种子、数据划分、支持集、权重哈希和嵌入身份。旧测试患者参与过训练的权重不能冒充该基准。

### 2. 单独运行一个对照

```powershell
python run_brain_novelty.py train `
  --manifest metadata/patient_split/split_manifest.jsonl `
  --support metadata/patient_split/support_manifest.json `
  --baseline-provenance runs/doc1027/baseline/baseline_provenance.json `
  --config configs/doc1027_staged/feature_synthetic.json `
  --device cuda --output runs/doc1027/feature_synthetic
```

替换配置及独立输出目录即可运行其他对照。当前六组均训练视觉适配器，**不要传 `--cache` 使用静态视觉特征缓存**。中断后使用同一身份和配置增加 `--resume`；实现、数据或配置发生改变时，精确恢复可能被拒绝，应使用新的实验目录。

### 3. 依次执行六个对照

```powershell
.\run_doc1027_stages.ps1 `
  -Manifest metadata/patient_split/split_manifest.jsonl `
  -Support metadata/patient_split/support_manifest.json `
  -BaselineProvenance runs/doc1027/baseline/baseline_provenance.json `
  -OutputRoot runs/doc1027_staged -Device cuda
```

这是本次上传新增的便携顺序启动脚本，调用现有训练入口；每组使用独立进程重新加载同一基准，可能比原本机复用编码器的控制脚本慢。它不在导入时启动训练，也不自动启动固定 σ、多种子或融合扩展。任一组失败即停止。

`-Stages feature_synthetic,image_synthetic` 可选择部分组；`-PrintOnly` 只显示启动参数；续跑指定组可增加 `-Resume`，但该组必须已存在可恢复的 checkpoint。基准需提前完成，未提供数据时脚本不能直接训练。

### 4. 汇总指标

```powershell
python report_brain_novelty.py --root runs/doc1027 --output runs/doc1027/summary.md
```

`--root` 指向需要汇总的共同父目录。每组保留 `config.json`、`identity.json`、训练日志、验证选模记录、checkpoint、`protocol.json`、`results.json`、`completion.json`；逐图分数和二值预测见 `evaluation/`，动态合成预览见 `synthetic_preview/`。病例热图的历史私有数据及本机专用控制脚本不随仓库分发。

## 本轮结果及边界

六组 seed=0 训练已完成。固定测试主要结果如下，数值为百分比：

| 方法 | Pixel AP ↑ | Dice 微平均 ↑ | 小病灶召回 ↑ | 正常误报面积 ↓ |
|---|---:|---:|---:|---:|
| 第一步基准 | 27.1202 | 37.1712 | 43.5897 | 7.2660 |
| 仅密集头 | 17.6478 | 34.9724 | 44.2308 | 5.8539 |
| 取样比例对照 | 19.1212 | 35.4995 | 44.8718 | 6.0906 |
| 特征伪异常 | 19.8876 | 31.1200 | 44.8718 | 5.8551 |
| 图像伪异常 | 13.1059 | 23.0934 | 57.0513 | 13.1535 |
| 两类联合 | 13.5525 | 24.9418 | 49.3590 | 4.6710 |
| 联合加排序 | 14.8079 | 24.3595 | 46.1538 | 4.7438 |

当前设置未超过第一步基准的整体定位表现。开发扩展门槛要求源验证 AP 同时超过密集头和第一步基准，并对二者均满足小病灶召回下降不超过2个百分点、正常误报面积增加不超过 max(相对10%，绝对0.1个百分点)。没有候选通过，因此固定 σ、seed1/2 和校准融合没有执行。

这是一组患者独立的已见胸腺瘤 CT 实验。测试集此前已被查看，不是全新盲测；缺独立未知疾病队列，不能声称验证了未知疾病泛化。MRI 元数据及划分仍需核实，不能直接将 CT 结果视作 MRI 结论。

## 自检与源码追溯

```powershell
python -m unittest discover -s paclip/diagnostics -p "test_*.py"
python prepare_patient_splits.py --help
python run_brain_novelty.py --help
python report_brain_novelty.py --help
```

单元测试使用生成的测试样本或替代编码器，不下载预训练主干，也不代表正式医学数据训练结果。上传前不重新启动完整训练。`SOURCE_MANIFEST.json` 保存原始源码和历史配置的 SHA-256；当前核心源码按原文件保存，新增内容为便携启动脚本、六组配置导出、依赖与说明文档。

上传前验证：115 项回归测试全部通过；六组导出配置与实际训练保存配置一致；全部 Python 源码语法、三个 Python 入口帮助命令以及 PowerShell 启动脚本的语法和仅预览模式均通过检查。
