# 模型市场（Model Hub）规范

replaygain-nyan 的模型共享机制：**本地训练 → 打包发布 → 别人浏览下载 → 直接写标签**。
训练/写标签永远在用户自己电脑上跑，hub 只负责分发模型包。

## 模型包格式（v3）

一个 zip，内含且仅含三类文件：

```
rg-model-{作者}-{yyyymmdd}.zip
├── meta.json            训练元数据（train_eval_export.py 自动生成）
├── MODEL_CARD.json      模型卡（统计 + opt-in 的歌单）
├── model.txt            单阶段模型，或
└── stage1/model_*.txt   两阶段一（42 维，LightGBM 原生格式）
    stage2/model_*.txt   两阶段二（42 + 组信号）
```

**只接受 LightGBM 原生 `.txt` 格式。** 旧版 pickle（.pkl）绝对不能分发——
pickle 加载时会执行任意代码，等于分发可执行文件。本地训练 v3 起自动产出生成原生格式。

## models.json 索引

hub 的入口是一个 JSON 数组（应用里用 `RG_HUB_URL` 环境变量指向它）：

```json
[
  {
    "id": "meow-citypop-20261020",
    "name": "rg-model-meow-20261020",
    "author": "meow",
    "date": "2026-10-20",
    "description": "city-pop 口味，整体压 -2dB 左右，慢歌更轻",
    "sample_count": 128,
    "mae": 0.71,
    "feature_pipeline_version": 2,
    "tracks_public": true,
    "download_url": "https://github.com/xxx/rg-models/releases/download/v1/rg-model-meow-20261020.zip",
    "sha256": "…zip 的 sha256…"
  }
]
```

字段说明：
- `feature_pipeline_version` 必须与发布时 train_eval_export.py 里的 `CACHE_VERSION` 一致；
  下载端会与本地比对，不符会拒绝加载（防静默劣化）
- `sha256` 为 zip 文件摘要，下载后强校验
- `mae` 单位 dB，越小越贴合训练者的标注

## 怎么发布

1. 本地：③ 训练页 → 训练完成 → 填署名/备注 → 「打包模型」下载 zip
2. 把 zip 传到本仓库的 GitHub Release（新增一个 tag，asset 传 zip）
3. 在 `models.json` 里加一条（PR 提交），审核通过后合并
4. 审核要点：zip 能解压、仅含 .txt/meta.json/MODEL_CARD.json、sha256 与索引一致、
   feature_pipeline_version 与当前主干一致、MODEL_CARD 完整

## 安全模型

- 应用加载 hub 模型时**只走 LightGBM 原生格式解析器**，绝不 joblib.load
- zip 解压前做路径校验（拒绝绝对路径 / `..`）和文件白名单（.txt / meta.json / MODEL_CARD.json）
- sha256 不匹配的包直接拒绝
- 以上都绕不过人工审核这一关：PR 就是审核队列
