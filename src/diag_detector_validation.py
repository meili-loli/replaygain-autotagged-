# -*- coding: utf-8 -*-
"""验证「标注打滑检测器」的召回率：往标签里注入已知噪声，看能不能找出来。

原理
----
检测器 = 「残差大」且「自己标签偏离特征邻域中位数」。
这里人为把一批歌的标签打歪，再跑检测器，看 Top-K 命中多少。
对照组（不注入）用来估计误报基线。

指标解读
  - 召回率 = 注入的假噪声里，被 Top-K 抓到的比例
  - 精确率 = Top-K 里，真的是被注入的比例
  - 「相对随机」= 命中率 / 随机抽取的期望命中率

实测结论（原作者数据）：注入 ±2~3.5 dB 时命中约 60%（相对随机 ≈ 9 倍），
注入 ±1.5~2.5 dB 时约 35%（≈ 5 倍）。**小于 2 dB 的打滑基本检测不出来。**

用法
  python src/diag_detector_validation.py
  python src/diag_detector_validation.py --cache handcraft_cache_42d_aug05.pkl --orig 909
"""
import argparse
import os
import sys
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
K_NB = 11
RESID_THR = 2.0
DEV_THR = 1.0


def reg(A, b, seed=0, n_jobs=8):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          min_child_samples=10, random_state=seed,
                          verbose=-1, n_jobs=n_jobs,
                          bagging_fraction=0.9, bagging_freq=1,
                          feature_fraction=0.9)
    m.fit(A, b)
    return m


def oof_original(X, yy, n_orig, groups, seeds=(0, 1, 2), n_jobs=8):
    """只产出「原始样本」的 OOF 预测（增强副本锁在同一折）。"""
    acc = np.zeros(n_orig)
    cnt = np.zeros(n_orig)
    for sd in seeds:
        for tr, va in GroupKFold(5).split(X, yy, groups):
            mask = va < n_orig
            if mask.sum() == 0:
                continue
            got = va[mask]
            acc[got] += reg(X[tr], yy[tr], sd, n_jobs).predict(X[got])
            cnt[got] += 1
    return acc / np.maximum(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache',
                    default=os.path.join(BASE, 'handcraft_cache_42d_aug05.pkl'),
                    help='含增强样本的缓存（前 N_ORIG 行是原始样本）')
    ap.add_argument('--v2-cache',
                    default=os.path.join(BASE, 'handcraft_cache_42d_v2.pkl'),
                    help='无增强缓存；找不到含增强缓存时用它，此时 groups 退化为每行一组')
    ap.add_argument('--orig', type=int, default=909, help='原始样本行数')
    ap.add_argument('--k', type=int, default=59, help='取检测分数前 K 名')
    ap.add_argument('--n-jobs', type=int, default=8)
    args = ap.parse_args()

    if os.path.exists(args.cache):
        c = joblib.load(args.cache)
        X = np.asarray(c['X'], np.float32)
        y0 = np.asarray(c['y'], np.float32)
        n_orig = min(args.orig, len(y0))
        if len(y0) == n_orig:
            groups = np.arange(n_orig)
            print('[INFO] 该缓存没有增强副本，groups 退化为每行一组')
        else:
            groups = np.tile(np.arange(n_orig), len(y0) // n_orig)
        print(f'缓存: {os.path.basename(args.cache)}  '
              f'X={X.shape}  原始 {n_orig} 首')
    elif os.path.exists(args.v2_cache):
        c = joblib.load(args.v2_cache)
        X = np.asarray(c['X'], np.float32)
        y0 = np.asarray(c['y'], np.float32)
        n_orig = len(y0)
        groups = np.arange(n_orig)
        print(f'缓存: {os.path.basename(args.v2_cache)}（无增强）'
              f'  X={X.shape}  {n_orig} 首')
    else:
        sys.exit(f'找不到缓存。先用 train_eval_export.py 生成 '
                 f'{os.path.basename(args.v2_cache)}')

    Xo = X[:n_orig]
    mu, sd_ = Xo.mean(0), Xo.std(0) + 1e-9
    nn = NearestNeighbors(n_neighbors=K_NB).fit((Xo - mu) / sd_)
    dist, ind = nn.kneighbors((Xo - mu) / sd_)

    def detect(yy):
        pred = oof_original(X, yy, n_orig, groups, n_jobs=args.n_jobs)
        ares = np.abs(yy[:n_orig] - pred)
        wmed = np.array([np.median(yy[:n_orig][ind[i, 1:]]) for i in range(n_orig)])
        dev = np.abs(yy[:n_orig] - wmed)
        return (ares >= RESID_THR) & (dev >= DEV_THR), ares, dev

    def run(tag, yy, victims):
        flag, ares, dev = detect(yy)
        idx = np.where(flag)[0]
        order = idx[np.argsort(-(ares * dev)[idx])]
        top = set(order[:args.k].tolist())
        hit = len(top & set(np.asarray(victims).tolist()))
        rec = 100 * hit / max(len(victims), 1)
        prec = 100 * hit / max(len(top), 1)
        exp = 100 * len(victims) / n_orig
        print(f'{tag:<40} 标记={flag.sum():3d}  Top{args.k}命中={hit:3d}'
              f'/{len(victims):3d}  召回={rec:5.1f}%  精确={prec:5.1f}%'
              f'  相对随机={rec/max(exp,1e-9):5.1f}x')
        return flag.sum(), hit

    print()
    print('=' * 96)
    print('对照组：不注入任何噪声（这些是检测器的误报）')
    print('=' * 96)
    rng = np.random.RandomState(0)
    run('对照组', y0.copy(), np.array([], int))

    print()
    print('=' * 96)
    print('实验组 A：注入 60 首 ±2~3.5 dB（大幅度打滑）')
    print('=' * 96)
    for trial in range(3):
        vic = rng.choice(n_orig, min(60, n_orig), replace=False)
        yc = y0.copy()
        yc[vic] += rng.choice([-1.0, 1.0], len(vic)) * rng.uniform(2.0, 3.5, len(vic))
        run(f'trial{trial+1}  注入 ±2~3.5dB', yc, vic)

    print()
    print('=' * 96)
    print('实验组 B：注入 60 首 ±1.5~2.5 dB（细微打滑，更接近真实情况）')
    print('=' * 96)
    for trial in range(2):
        vic = rng.choice(n_orig, min(60, n_orig), replace=False)
        yc = y0.copy()
        yc[vic] += rng.choice([-1.0, 1.0], len(vic)) * rng.uniform(1.5, 2.5, len(vic))
        run(f'trial{trial+1}  注入 ±1.5~2.5dB', yc, vic)


if __name__ == '__main__':
    main()
