# 🐱 ReplayGain-nyan

> 「让一千首歌的响度，都乖顺得像猫猫被摸头一样～」ฅ(=ↀωↀ=)ฅ

**ReplayGain-nyan** 是一个专门给你的私有音乐库自动打 ReplayGain 标签的小工具。
它不靠固定的 −18 LUFS 死阈值，而是**学你自己的耳朵**：你标过哪几首"该衰减多少"，
它就学会"什么样的歌该衰减多少"，然后对着全库一首一首地贴标签。

> 主人只要标个几十首就够了，剩下的交给猫猫！喵～ (｡•̀ᴗ-)✧

---

## ✨ 它到底会做什么

| | 功能 | 一句话 |
|---|---|---|
| 🔍 | **自动提取特征** | 42 维响度 / 频谱 / 动态特征，ffmpeg 解码，不吃内存 |
| 🧠 | **学习你的听感** | LightGBM 回归，学的是「这首歌你希望衰减多少 dB」 |
| 🏷️ | **批量打标签** | 直接写 `REPLAYGAIN_TRACK_GAIN` / `_PEAK`，foobar2000 立刻认 |
| 🎧 | **内置标注器** | 带 A/B 试听、精确响度对齐的 PyQt 标注工具，标一首只花你 20 秒 |
| 🩺 | **标签体检** | 用特征邻域找出「标注时手滑了」的歌，只让你复核那一小撮 |
| 🎯 | **组信号两阶段** | 同专辑 / 同艺人的预测当特征，猫猫会看着邻居来决定✨ |

---

## 📊 实测成绩（867 首，10 seed 交叉验证）

猫猫不说大话，这是真·干净协议下的 OOF 成绩喵：

| 模型 | MAE | \|误差\| > 2 dB | \|误差\| > 3 dB | 最大误差 |
|---|---:|---:|---:|---:|
| 双专家 + 门控（原始架构） | 0.940 | 95 首 | 63 首 | 5.94 |
| 单 LGB 回归（42 维） | 0.688 | 16 首 | 5 首 | 4.34 |
| **本仓库最终版（42 维 + 组信号）** | **0.666** | **10 首** | **3 首** | **3.80** |

> 🎀 以「误差 ≤ 0.75 dB 就等于无误差」（人耳分辨不了）计，
> 现在有 **64.5%** 的歌落在这一档里，而**从未被人耳验证过**的那 808 首
> 更是有 **66.3%** 达标 (`=^･ω･^=`)

**组信号这一招到底值多少？** 逐 seed 配对胜负 **10/10**，不是运气喵：

| 增量特征 | MAE | >2 dB | 配对胜率 |
|---|---:|---:|---:|
| 无（42 维） | 0.688 | 16 | — |
| + 目录（2 维） | 0.680 | 14 | 10/10 |
| + 专辑 tag（2 维） | 0.673 | 13 | 10/10 |
| + 艺人 tag（2 维） | 0.678 | 12 | 8/10 |
| **+ 目录 + 专辑 + 艺人（6 维）** | **0.666** | **10** | **10/10** |

---

## 🚀 快速开始

```bash
pip install -r requirements.txt
```

需要系统里有 `ffmpeg` 和 `ffprobe`（PATH 能被找到就行）。

### 1️⃣ 先标注一小撮歌，教猫猫认识你的耳朵

```bash
python src/annotate_gain.py --input "/path/to/one/album"
```

界面快捷键：`Space` 播放/暂停 · `↑↓` 微调 ±0.5 dB · `T` A/B 对比 · `←→` ±5 秒 · `S` 保存下一首

导出的 `delta_gain.csv` 就是你教给猫猫的全部教材：
```csv
file,delta_gain
song-a.flac,-9.5
song-b.flac,-11.0
```

### 2️⃣ 训练

```bash
# 基础版（单阶段，42 维）
python src/train_eval_export.py \
    --csv delta_gain.csv \
    --audio-root "/path/to/your/music" \
    --n-jobs 8 --shift -5

# 加上组信号（推荐，需要先建元数据）
python src/build_library_meta.py --root "/path/to/your/music"
python src/train_eval_export.py \
    --csv delta_gain.csv \
    --audio-root "/path/to/your/music" \
    --n-jobs 8 --shift -5 --group-signal
```

### 3️⃣ 贴标签

```bash
# 先空跑，只看不写
python src/predict_and_tag.py --input "/path/to/your/music" --dry-run --workers 6

# 确认无误再真写
python src/predict_and_tag.py --input "/path/to/your/music" --workers 6
```

> ⚠️ **`--workers` 不要超过 8**。librosa 的 HPSS 很吃内存，
> 实测 16 个 worker 会在 31 GB 内存的机器上直接 MemoryError 喵… (´･_･`)
>
> ⚠️ **组信号模式要批量跑全库**（至少整张专辑）。只跑几十首时
> 「专辑/艺人」组里只有它自己一个成员，算不出信号，模型会默默退化成 42 维行为。

---

## 🧰 主要脚本

| 脚本 | 作用 |
|---|---|
| `src/annotate_gain.py` | PyQt 标注器，支持 `--candidates` 只复核指定清单、`--blind` 盲测 |
| `src/train_eval_export.py` | 训练 + OOF 评估 + 导出模型；`--group-signal` 开两阶段 |
| `src/predict_and_tag.py` | 批量推理并写标签；支持 `two_stage_lgb` 模型 |
| `src/find_label_outliers.py` | 揪出疑似「标注手滑」的歌曲清单 |
| `src/group_signal.py` | 组信号特征（**训练与推理共用同一份实现**） |
| `src/build_library_meta.py` | 从音频 tag / 文件名建全库的目录 / 专辑 / 艺人表 |
| `src/compare_tags.py` | 新预测 vs 文件里现有标签的对比 |
| `src/label_confidence.py` | 给标签打可信度分级，修正评估口径 |
| `src/migrate_feat_cache.py` | 音乐库搬家后按文件名迁移特征缓存 |
| `src/diag_honest_eval.py` | 无泄漏评估协议（量化 OOF 泄漏量） |
| `src/diag_detector_validation.py` | 用注入噪声验证「打滑检测器」的召回率 |

---

## 🐾 猫猫踩过的坑（都写在这里了，主人别再踩）

这些是花了一整天、几十个实验换来的，每一条都有数据支撑喵：

### 1. 朴素 KFold + 数据增强 = 评估泄漏 ⚠️

把 909 原始 + 909 增强样本堆成 1818 行再随机切分，同一首歌的增强副本漏进了训练集。
**虚报 MAE 0.383，真实是 0.940**。要用 `GroupKFold` 锁同折。

### 2. 报"5 模型平均"的分数是骗自己

`meta.json` 里没有 `n_ensemble` 时 `load_models()` 默认 1，实际部署的是单模型。
**训练时的 OOF 分数必须在「推断时真正会用的那套模型」上测。**

### 3. `random_state` 对 LightGBM 是空操作

不设 `bagging_fraction` / `feature_fraction` 时，`seed=0` 和 `seed=1` 的预测**逐位相同**。
`--n-ensemble 5` 训出来的是 5 个一模一样的模型，收益恒为 0。

### 4. 后处理校准基本都不泛化

Isotonic / 全局线性 / 分组平移 / 保序校准……全试过，**没有一个在折外站得住**。
原因：偏置是「预测值」的函数，而预测值本身含噪声。
**任何后处理都要多 seed 验证**，单 seed 的漂亮数字多半是折划分的运气。

### 5. 双专家 / 分专家架构是负收益

不管软路由还是硬路由，都不如一个单回归。连 Ridge 线性回归都比它好。
（根因是 `-10 <= y < -9` 那段两个专家都没训过的空洞。）

### 6. 「同专辑」不能当规则，只能当特征 🐈

拿「同专辑标签中位数」当规则去建议，实测三处**全错**。
但把它做成「组内其他成员的预测均值」这个**特征**喂给模型，10/10 稳定获胜。
> 血泪教训：位置/分组信息当规则是拍脑袋，当特征是让模型自己学。

### 7. 响度不变性增强：物理上成立，训练上灾难

给音频加 +Δ dB、标签也 +Δ，物理完全自洽（实测 5 维精确 +Δ、33 维精确不变）。
但模型学到的不是「偏移 Δ」，而是「同一首歌的多个副本标签不同 → 回归均值」。
**MAE 0.688 → 0.811，>2 dB 从 16 首涨到 40 首**。

### 8. OOF 误差² = 模型真实误差² + 标签噪声²

对已有标签测出来的误差**永远 ≥ 标签噪声**。
人耳噪声 0.5 dB 就决定了 MAE 下不到 ~0.40。
**所以别拿 MAE 当唯一目标**——先问「多少误差是人耳分辨不出的」（本项目用 0.75 dB）。

---

## 📁 目录结构

```
replaygain-nyan/
├── src/                         # 全部可执行脚本
│   ├── annotate_gain.py         # 标注器（PyQt）
│   ├── train_eval_export.py     # 训练 + 评估 + 导出
│   ├── predict_and_tag.py       # 批量推理写标签
│   ├── group_signal.py          # 组信号（训练/推理共用）
│   ├── build_library_meta.py    # 全库元数据
│   ├── find_label_outliers.py   # 标签打滑检测
│   ├── label_confidence.py      # 标签可信度分级
│   ├── compare_tags.py          # 新旧标签对比
│   ├── migrate_feat_cache.py    # 特征缓存迁移
│   ├── diag_honest_eval.py      # 无泄漏评估
│   ├── diag_detector_validation.py
│   └── validate_two_stage.py    # 两阶段端到端验证
├── docs/
│   └── 诊断_模型真实精度.md       # 完整诊断报告（含全部实验数据）
├── requirements.txt
└── .gitignore                   # 已经把个人数据全挡在外面了喵
```

运行时会在**当前工作目录**生成这些：

```
delta_gain.csv                  # 你的标注（教材）
feat_cache/                     # 42 维特征缓存，key = 绝对路径的 md5
handcraft_cache_42d_v2.pkl      # 汇总缓存
library_meta.csv                # 目录/专辑/艺人表
model_dir/stage1/ stage2/       # 两阶段模型
eval_out/                       # OOF 评估产物
inference_logs/                 # 每次推理的日志
```

---

## 🎯 `--shift` 是干嘛的

如果你习惯把所有 `delta_gain` 统一减去一个常数（本项目作者是减 5），
用 `--shift -5` 让分组统计和阈值也跟着下移。这个值会写进 `meta.json`，
推理时自动读取，**不用每次手动传**喵。

---

## 📜 许可证

MIT。随便用、随便改，但猫猫不对你听感负责喵 (=･ｪ･=)

---

<div align="center">

**ฅ^•ﻌ•^ฅ**　「响度卷成一条线之后，世界就安静了喵～」

</div>
