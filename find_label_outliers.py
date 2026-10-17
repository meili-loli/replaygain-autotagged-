# -*- coding: utf-8 -*-
"""
找出「疑似标注打滑」的歌 —— 不用重标全部，只复核这一小撮。

原理
----
一首歌如果：
  (a) 模型预测和它的标签差很多            -> 残差大
  (b) 音频特征接近的邻居们标签却很一致     -> 邻域中位数可信
  (c) 它自己的标签偏离邻域中位数很远       -> 它自己孤立
那它大概率是标注时手滑了，而不是模型学不会。

有效性（注入已知噪声验证，见 diag_detector_validation.py）
  - 注入 ±2~3.5 dB：Top59 能找回 ~60%，相对随机提升约 9 倍
  - 注入 ±1.5~2.5 dB：召回 ~35%，提升约 5 倍
  - 小于 2 dB 的打滑基本检测不出来
  所以清单里大约 6 成是真错，另外 4 成你听了会说「没错」。

数据来源
  eval_out/per_idx.csv          训练脚本产出的 OOF 预测（就是模型自己的残差）
  handcraft_cache_42d_v2.pkl    特征 + 真实路径

用法
  python find_label_outliers.py
  python find_label_outliers.py --min-resid 2.5     # 收紧门槛，清单更短更准
  python find_label_outliers.py --top 40
"""

import os
import io
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CACHE = 'handcraft_cache_42d_v2.pkl'
PER_IDX = os.path.join('eval_out', 'per_idx.csv')
N_NEIGHBORS = 11


def neighbourhood(X, y, k=N_NEIGHBORS):
    """标准化特征空间里的邻居标签统计（距离加权中位数）。"""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    dist, ind = nn.kneighbors(Z)

    n = len(y)
    wmed = np.zeros(n)
    spread = np.zeros(n)
    nn_dist = np.zeros(n)
    for i in range(n):
        nb = ind[i, 1:]
        d = dist[i, 1:]
        w = 1.0 / (d + 1e-6)
        order = np.argsort(y[nb])
        cw = np.cumsum(w[order]) / w.sum()
        wmed[i] = y[nb][order][np.searchsorted(cw, 0.5)]
        spread[i] = np.median(np.abs(y[nb] - np.median(y[nb])))
        nn_dist[i] = np.median(d)
    return wmed, spread, nn_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=0, help='只输出前 N 首（0=全部）')
    ap.add_argument('--min-resid', type=float, default=2.0,
                    help='残差门槛，默认 2.0 dB')
    ap.add_argument('--min-dev', type=float, default=1.0,
                    help='偏离邻域中位数的门槛，默认 1.0 dB')
    ap.add_argument('--per-idx', default=PER_IDX)
    ap.add_argument('--cache', default=CACHE)
    ap.add_argument('--out-suffix', default='',
                    help='输出文件名后缀，如 _round2 -> review_candidates_round2.csv')
    args = ap.parse_args()

    if not os.path.exists(args.per_idx):
        sys.exit(f'找不到 {args.per_idx}，先跑一次 train_eval_export.py')

    c = joblib.load(args.cache)
    X = np.asarray(c['X'], np.float32)
    y = np.asarray(c['y'], np.float32)
    paths = c['orig_paths']

    grp = pd.read_csv(args.per_idx)
    grp = grp.sort_values('idx').reset_index(drop=True)
    if len(grp) != len(y) or not np.allclose(grp['y'].values, y, atol=1e-4):
        print('[WARN] per_idx.csv 与缓存对不上，按 idx 对齐')
        grp = grp.set_index('idx').reindex(range(len(y))).reset_index()
        y = grp['y'].values.astype(np.float32)

    pred = grp['pred'].values.astype(np.float64)
    resid = y - pred
    ares = np.abs(resid)
    print(f'样本 {len(y)} 首   OOF MAE={ares.mean():.3f}   '
          f'>2dB={int((ares > 2).sum())} 首   max={ares.max():.2f}')

    print('计算特征空间邻域标签统计 ...')
    wmed, spread, nndist = neighbourhood(X, y)
    loc_dev = np.abs(y - wmed)

    df = pd.DataFrame({
        'idx': np.arange(len(y)),
        'file': [os.path.basename(p) for p in paths],
        'dir': [os.path.dirname(p) for p in paths],
        'y_label': np.round(y, 2),
        'y_pred': np.round(pred, 2),
        'resid': np.round(resid, 2),
        'abs_resid': np.round(ares, 2),
        'nn_median': np.round(wmed, 2),
        'dev_from_nn': np.round(loc_dev, 2),
        'nn_spread': np.round(spread, 2),
        'nn_dist': np.round(nndist, 3),
        'suggest': np.round(0.5 * (pred + wmed) * 2) / 2,   # 折中值，取 0.5 网格
    })

    mask = (df.abs_resid >= args.min_resid) & (df.dev_from_nn >= args.min_dev)
    cand = df[mask].copy()
    cand['_score'] = cand.abs_resid * cand.dev_from_nn / (1.0 + cand.nn_spread)
    cand = cand.sort_values('_score', ascending=False).drop(columns='_score')
    hard = df[~mask & (df.abs_resid >= args.min_resid)] \
        .sort_values('abs_resid', ascending=False)

    sfx = args.out_suffix
    cand_p = f'review_candidates{sfx}.csv'
    hard_p = f'review_hard{sfx}.csv'
    cand.to_csv(cand_p, index=False, encoding='utf-8-sig')
    hard.to_csv(hard_p, index=False, encoding='utf-8-sig')

    print()
    print('=' * 78)
    print(f'[疑似标注打滑] {len(cand):4d} 首  -> {cand_p}')
    print(f'   残差大 且 自己标签偏离邻域中位数 >= {args.min_dev} dB')
    print(f'[真·难样本]    {len(hard):4d} 首  -> {hard_p}')
    print(f'   残差大 但邻居标签也乱 -> 模型在这片区域学不会，不是标注的错')
    print('=' * 78)

    show = cand.head(args.top) if args.top > 0 else cand.head(20)
    if len(show):
        print(f'{"文件":<46}{"你的标":>7}{"模型":>7}{"邻居":>7}{"建议":>7}{"差":>7}')
        print('-' * 78)
        for _, r in show.iterrows():
            print(f'{r["file"][:44]:<46}{r.y_label:>7.2f}{r.y_pred:>7.2f}'
                  f'{r.nn_median:>7.2f}{r.suggest:>7.2f}{r.resid:>+7.2f}')
        print()
        print(f'预计复核 {len(cand)} 首（约 {max(1, len(cand) * 25 // 60)} 分钟）')


if __name__ == '__main__':
    main()
