# K2G-BiCTC连续手语识别

K2G-BiCTC（Keypoints-to-Gloss + BiLSTM+CTC）是一个句级连续手语识别项目，采用
`MediaPipe 关键点 + BiLSTM + CTC` 的技术路线，并按 CE-CSL 风格组织数据。

本仓库目标是手语识别（gloss 序列预测）。

## 功能特性

- 生成 CE-CSL 的 `manifest` 与词表
- 基于 MediaPipe Hands + Pose 的视频关键点预处理
- BiLSTM+CTC 训练流程
- 离线视频推理
- 实时摄像头推理

## 仓库结构

```text
K2G-BiCTC/
  configs/
    bilstm_ctc.yaml
  data/
    README.md
    raw/                # 已被 git 忽略，放原始数据
    meta/               # 已被 git 忽略，放 manifest/vocab
    processed/          # 已被 git 忽略，放关键点 npz
  outputs/              # 已被 git 忽略，放 checkpoint 和推理输出
  scripts/
    prepare_cecsl.py
    extract_keypoints.py
    train_ctc.py
    infer_ctc.py
  src/bisignlangtrans/
    data/
    models/
    decoding.py
```

## 环境要求

- Python 3.10+
- Linux/macOS/Windows（摄像头模式需要 OpenCV GUI 支持）

安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

## 数据集准备

本仓库不包含 CE-CSL 数据，请将数据放到以下目录：

```text
data/raw/CE-CSL/
  label/
    train.csv
    dev.csv
    test.csv
  video/
    train/<Translator>/*.mp4
    dev/<Translator>/*.mp4
    test/<Translator>/*.mp4
```

详细说明见 `data/README.md`。

## 训练流程

1. 生成 `manifest.jsonl` 和 `vocab.json`：

```bash
python scripts/prepare_cecsl.py \
  --raw-root data/raw/CE-CSL \
  --manifest data/meta/manifest.jsonl \
  --vocab data/meta/vocab.json \
  --min-freq 2
```

2. 提取关键点到 `data/processed/*.npz`：

```bash
python scripts/extract_keypoints.py \
  --manifest data/meta/manifest.jsonl \
  --processed-root data/processed \
  --num-workers 6 \
  --chunk-size 32
```

3. 训练 BiLSTM+CTC：

```bash
python scripts/train_ctc.py --config configs/bilstm_ctc.yaml
```

最佳模型默认保存到：

```text
outputs/checkpoints/best.pt
```

## 推理

视频推理：

```bash
python scripts/infer_ctc.py \
  --video data/raw/CE-CSL/video/test/A/test-00001.mp4 \
  --checkpoint outputs/checkpoints/best.pt \
  --output-json outputs/infer/test-00001.json
```

摄像头推理：

```bash
python scripts/infer_ctc.py \
  --camera-id 0 \
  --checkpoint outputs/checkpoints/best.pt \
  --camera-min-frames 24 \
  --camera-infer-every 6 \
  --camera-window-frames 96
```

## 复现说明

- 主要超参数在 `configs/bilstm_ctc.yaml`
- 训练摘要默认保存为 `outputs/checkpoints/summary.json`
- 默认解码方式是 CTC greedy decode

## 许可证

MIT，详见 `LICENSE`。

## 致谢

- CE-CSL 数据集维护者
- MediaPipe 与 PyTorch 社区
