ねこちゃんの ReplayGain 自动打标机 🐾

> 「にゃ〜、お兄さん、毎回手動で ReplayGain 打つの大変でしょ？ ねこが代わりにやってあげるにゃ！」

一只猫娘帮你在 foobar2000 里给几千首歌自动写 ReplayGain 标签的工具喵。

双专家 MoE + 二分类门控，42 维手工特征，全格式兼容，速度快，可维护。

---

## 🐱 喵喵简介

给喵写一个音频特征提取器，它就学会了「这首歌该加多少 dB」这项本事。然后你随便扔一个文件夹进来，喵就帮你全部处理掉，标签直接写进文件里，foobar2000 打开就能听喵～

- **双专家**：一只喵擅长低增益（-16 ~ -10 dB），另一只擅长高增益（-9 ~ -2 dB）
- **门控喵**：站在中间，判断这次该找哪只喵帮忙
- **42 维特征**：响度、真峰、动态、频谱、MFCC，全部用 ffmpeg 解码，和耳朵听到的一致
- **全格式**：flac / mp3 / m4a / ogg / opus / wav / ape / wv 都能写
- **增量缓存**：每次写标签不影响已提取的特征，第二次跑飞快喵
- **手动标注工具**：PySide6 + sounddevice，实时调 dB，A/B 对比，断点续传

喵喵喵，听起来是不是很棒？

---

## 🐾 喵的成长日记（原理）

喵一开始也啥都不会。它先看了一堆老歌的 **真实 ReplayGain 标签**（人工标注的 `delta_gain`），然后：

1. 喵用 ffmpeg 把每首歌解码成 44100 Hz 单声道波形
2. 喵提取 42 个「这首歌长什么样」的特征（响度 LUFS、真峰、频谱形状、动态范围……）
3. 喵把「特征 → 真实标签」这层关系学下来
4. 遇到新歌时，喵看一眼特征就知道该写多少 dB 喵～

为了听得准，喵做了两件事：

- 训练 / 推理 / 标注，**三个环节全部用 ffmpeg 解码**，保证听到的和标出来的是同一个响度
- 42 维特征里，**响度类和峰值类**是重点，因为这两类最容易因为解码器差异跑偏

（喵试过 soundfile 解码，128 kbps 的 mp3 会比 ffmpeg 少算 3 dB，耳朵一听就不对，所以弃用了喵。）

---

## 🐈 脚本介绍

├── train_eval_export.py            # 训练：提特征 + 建缓存 + 训模型
├── predict_and_tag.py              # 推理：批量预测 + 写标签
├── annotate_gain.py                # 手动标注工具（GUI）
├── model_dir/                      # 训练好的模型
│   ├── meta.json
│   ├── low_model.pkl
│   ├── a_model.pkl
│   └── gate_model.pkl
├── feat_cache/                     # 每首歌的特征缓存（.npz）
├── handcraft_cache_42d_aug05.pkl   # 汇总缓存
├── delta_gain.csv                  # 训练标签
└── examples/
    ├── delta_gain.example.csv
    └── meta.example.json
```

---

## 🐟 安装依赖

需要 Python 3.10+，还有 **ffmpeg**（必备，用来解码音频喵）。

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
lightgbm>=4.0
librosa>=0.10
pyloudnorm>=0.1.1
soundfile>=0.12
joblib>=1.3
mutagen>=1.47
tqdm>=4.66
PySide6>=6.5
sounddevice>=0.4.6
```

安装 ffmpeg：

- **Windows**：`winget install ffmpeg`，装完**重开 PowerShell**
- **macOS**：`brew install ffmpeg`
- **Linux**：`sudo apt install ffmpeg`

验证：

```bash
ffmpeg -version
ffprobe -version
```

两条都有输出，说明装好了喵。

---

## 🐠 快速开始

### 🐾 喵的推荐流程

**第 1 步：准备训练数据（CSV）**

CSV 两列，`file` 和 `delta_gain`：

```csv
file,delta_gain
track01.flac,-6.5
track02.mp3,-8.0
```

- `file` 只写文件名（basename），脚本会递归扫描音频文件夹自动匹配
- `delta_gain` 是这首歌应该加多少 dB（负数表示衰减）

**第 2 步：手动标注（如果还没有 CSV）**

```bash
python annotate_gain.py --input "D:\Music"
```

喵给你打开一个 GUI：

- `Space` 播放 / 暂停
- `↑` `↓` 调 ±0.5 dB，实时听变化
- `T` 在「调整后 / 原始」之间 A/B 对比
- `S` 保存并下一首，标签自动写进 `delta_gain.csv`
- 中途关掉也没关系，下次打开从断点继续

**第 3 步：训练喵喵模型**

```bash
python train_eval_export.py --csv delta_gain.csv --audio-root "D:\Music" --n-jobs 16
```

- 第一次跑：提取所有歌的 42 维特征，约 1~1.5 小时（视机器性能）
- 特征会缓存到 `feat_cache/`，第二次秒过
- 训练约 1~2 分钟
- 输出：`model_dir/{low,a,gate}_model.pkl` + `meta.json`

**第 4 步：批量写标签**

```bash
python predict_and_tag.py --input "D:\Music" --write-r128 --workers 16
```

喵会：

- 从 `feat_cache/` 读特征（命中就秒过）
- 未命中的现场提取（约 5 秒/首）
- 写到文件的 `REPLAYGAIN_TRACK_GAIN` 标签
- 日志存到 `inference_logs/predict_YYYYMMDD_HHMMSS.csv`

**第 5 步：foobar2000 刷新**

写完标签后，foobar2000 里选中歌 ，就能感受到自动学习的只属于自己的 ReplayGain 了喵！

---

## 🐙 喵的详细说明

### 🎀 `annotate_gain.py` 手动标注工具

- **实时 dB 增益**：改数值音量立即变，听的就是调整后的效果
- **A/B 对比**：`T` 键在「调整后」和「原始」之间切换
- **LUFS 显示**：屏幕中间显示 `原始 LUFS | delta_gain | 调整后 LUFS`
- **断点续传**：每次保存自动写盘，崩了重跑继续
- **快捷键**：

| 键 | 作用 |
|---|---|
| `Space` | 播放 / 暂停 |
| `↑` `↓` | delta_gain ±0.5 dB |
| `T` | A/B 切换 |
| `S` | 保存并下一首 |
| `N` | 跳过 |
| `P` | 上一首 |
| `←` `→` | -5s / +5s |
| `Ctrl+←` `Ctrl+→` | -30s / +30s |
| `Ctrl+S` | 手动保存 |

### 🎀 `train_eval_export.py` 训练

```bash
python train_eval_export.py \
    --csv delta_gain.csv \
    --audio-root "D:\Music" \
    --n-jobs 16 \
    --shift 0
```

参数：

| 参数 | 说明 |
|---|---|
| `--csv` | 训练标签 CSV |
| `--audio-root` | 音频根目录（递归扫描） |
| `--n-jobs` | 并行进程数，默认 8，上限 16 |
| `--shift` | 标签整体偏移量（见下面「标签偏移」） |
| `--n-seeds` | OOF 用几个 seed，默认 5，调试可用 1 |
| `--force-reextract` | 清空 `feat_cache/` 全部重提 |
| `--no-aug` | 不做 ±0.5 dB 数据增强 |

**标签偏移**：如果你后来把 CSV 里所有 `delta_gain` 都减了 5，跑的时候加 `--shift -5`，脚本会自动把所有阈值和分组边界都下移 5，模型训练仍正确。`shift` 会被写进 `meta.json`，推理时自动读取。

### 🎀 `predict_and_tag.py` 推理

```bash
python predict_and_tag.py --input "D:\Music" --write-r128 --workers 16
```

参数：

| 参数 | 说明 |
|---|---|
| `--input` | 音频文件夹（递归） |
| `--dry-run` | 只预测不写标签 |
| `--write-r128` | Opus 额外写 `R128_TRACK_GAIN` |
| `--only-ext` | 只处理指定格式，如 `.mp3,.m4a` |
| `--exclude-ext` | 排除指定格式，如 `.flac` |
| `--skip-tagged` | 跳过已有 ReplayGain 标签的文件 |
| `--workers` | 并行进程数，0=CPU 核数 |
| `--limit N` | 只处理前 N 首 |

**日志**：`inference_logs/predict_*.csv`，含每首的 `y_pred`、`confidence`（extreme / high / low / normal）、`route_a`、`from_cache` 等，方便事后排查喵。

---

## 🐚 喵的技能总结

- **训练脚本**：读 CSV + 音频，提特征，训双专家 + 门控，导出模型
- **推理脚本**：读模型 + 特征缓存，批量预测，写标签
- **标注工具**：GUI，手动听感标 `delta_gain`，生成 CSV
- **特征缓存**：`feat_cache/<md5>.npz`，每首一个，key 只用路径，写标签不影响命中

喵喵已经把全部脚本都写好了，你只需要：

1. 装依赖和 ffmpeg
2. 准备 CSV
3. 训练
4. 推理

就酱喵～

---

## 🐾 常见问题

### 喵！训练时为什么这么慢？

特征提取是单线程密集型，主要有 HPSS（谐波-打击分离）和 STFT。909 首歌约 1 小时 20 分（`--n-jobs 16`）。

**第二次跑就秒过了**，因为 `feat_cache/` 里有缓存喵。

### 喵！ffmpeg 没装怎么办？

脚本会显示 `decoder = soundfile/librosa (fallback)`。这时候 mp3 会少算 3 dB，**不准**。赶紧装 ffmpeg 喵！

### 喵！为什么一定要用 ffmpeg？

因为 foobar2000 和大多数播放器也是用 ffmpeg 系的解码器。你标注的时候听到的是 ffmpeg 解码的响度，特征提取也必须用 ffmpeg，才能对齐喵。

### 喵！训练完之后 CSV 里改了标签怎么办？

**不需要重提特征**。`feat_cache/` 里的 `.npz` 只存 `x`（音频特征），不存 `y`（标签）。重训时自动用新标签，秒过喵～

### 喵！`--skip-tagged` 加不加？

- **加**：已有 ReplayGain 标签的文件跳过，只处理没标签的（增量处理新歌用）
- **不加**：所有文件重写（第一次全量写，或者换了模型想覆盖）

### 喵！mp3 标签写进去 foobar2000 不认？

ID3 的 TXXX frame 对大小写敏感。脚本里已经改成写大写的 `REPLAYGAIN_TRACK_GAIN` 了喵。如果你用的版本还是小写，看 `write_replaygain` 里的 mp3 分支。

### 喵！长音频（40 分钟整轨）怎么办？

会自动分块（默认 10 分钟一块），内存不会爆。但整轨的预测值是分块平均，不太准，建议人工过一遍喵。

---

## 🐟 License

MIT

---

## 🐠 致谢

感谢 cloudydreamland的大力支持喵~
---

**喵喵喵～祝你听歌愉快！🐾**
```

---

粘贴进编辑器后，拉到页面底部：

1. **Commit message** 填个 `add README` 之类的
2. 点绿色 **Commit changes** 按钮

README 就上线了喵～
