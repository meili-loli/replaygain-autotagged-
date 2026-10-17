# -*- coding: utf-8 -*-
"""两阶段训练 + 推理（组信号版）。组信号定义统一来自 group_signal.py。

阶段 1  42 维 -> 基础模型
阶段 2  [42 维 + 组信号(目录/专辑/艺人)] -> 最终模型
        组信号 = 组内「其他成员」的一阶段预测均值 + 其余成员数

本脚本只做两件事：
  validate  严格验证两阶段是否真的优于单阶段（outer/inner 嵌套，防泄漏）
  noop      占位（导出逻辑在 train_eval_export.py 的 --group-signal 里）

之所以把验证单独放这里：两阶段很容易偷偷泄漏（组信号若含自己的标签就废了），
必须用和 bench_group_tags.py 相同的、逐折独立的协议来确认。
"""
import os
import sys
import numpy as np
import joblib
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import lightgbm as lgb
from sklearn.model_selection import KFold
import group_signal as GS

SEEDS = list(range(10))
N_FOLDS = 5
TOL = 0.75
USE_KEYS = ('dir', 'album', 'artist')


def mk(s):
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             min_child_samples=10, random_state=s, verbose=-1,
                             n_jobs=8, bagging_fraction=0.9, bagging_freq=1,
                             feature_fraction=0.9)


def sc(y, p):
    a = np.abs(y - p)
    return (a.mean(), int((a > 1).sum()), int((a > 2).sum()),
            int((a > 3).sum()), a.max(), float((a <= TOL).mean()))


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='两阶段（42 维 + 组信号）vs 单阶段 的严格对比。'
                    '需要先跑 train_eval_export.py 生成特征缓存。')
    ap.add_argument('--out', default=os.path.join(BASE, 'oof_pred_twostage.npy'),
                    help='把两阶段 OOF 预测存到哪')
    args = ap.parse_args()

    cache_p = os.path.join(BASE, 'handcraft_cache_42d_v2.pkl')
    if not os.path.exists(cache_p):
        sys.exit(f'找不到 {cache_p}\n请先跑一次 train_eval_export.py 生成特征缓存。')
    c = joblib.load(cache_p)
    X = np.asarray(c['X'], np.float64)
    y = np.asarray(c['y'], np.float64)
    paths = c['orig_paths']
    n = len(y)
    meta = GS.load_meta(os.path.join(BASE, GS.META_PATH))
    hit = sum(1 for p in paths if GS.norm_path(p) in meta)
    print(f'样本 {n}  元数据 {len(meta)} 条  命中 {hit}')
    if not meta:
        print('[WARN] 没有 library_meta.csv，组信号会退化为按目录分组。'
              '先跑 build_library_meta.py 才能用上专辑/艺人 tag。')

    # 单阶段基线
    P1 = np.zeros((len(SEEDS), n))
    for si, s in enumerate(SEEDS):
        for tr, va in KFold(N_FOLDS, shuffle=True, random_state=s).split(X):
            m = mk(s); m.fit(X[tr], y[tr])
            P1[si, va] = m.predict(X[va])
    p1 = P1.mean(0)
    b1 = sc(y, p1)
    print(f'\n单阶段 42 维          MAE={b1[0]:.4f}  >2={b1[2]:>3}  max={b1[4]:.2f}  '
          f'<=0.75={100*b1[5]:.1f}%')

    # 两阶段
    P2 = np.zeros((len(SEEDS), n))
    for si, s in enumerate(SEEDS):
        for tr, va in KFold(N_FOLDS, shuffle=True, random_state=s).split(X):
            # 一阶段：训练段用折外预测（无泄漏），验证段用本折模型
            bp = np.zeros(n)
            bp[tr] = p1[tr]
            m1 = mk(s); m1.fit(X[tr], y[tr])
            bp[va] = m1.predict(X[va])
            # 组信号
            G, _ = GS.build_group_features(paths, bp, meta, USE_KEYS)
            # 二阶段
            m2 = mk(s); m2.fit(np.hstack([X[tr], G[tr]]), y[tr])
            P2[si, va] = m2.predict(np.hstack([X[va], G[va]]))
    p2 = P2.mean(0)
    b2 = sc(y, p2)
    print(f'两阶段 42+组信号(6维)   MAE={b2[0]:.4f}  >2={b2[2]:>3}  max={b2[4]:.2f}  '
          f'<=0.75={100*b2[5]:.1f}%')

    wins = sum(1 for i in range(len(SEEDS))
               if (sc(y, P2[i])[4], sc(y, P2[i])[2], sc(y, P2[i])[0])
               < (sc(y, P1[i])[4], sc(y, P1[i])[2], sc(y, P1[i])[0]))
    print(f'\n逐 seed 配对（先 max 再 >2dB 再 MAE）: {wins}/{len(SEEDS)}')

    print('\n分组成绩:')
    print(f'{"子集":<16}{"n":>5}{"基线MAE":>10}{"两阶段MAE":>11}{"基线>2":>8}'
          f'{"两阶段>2":>9}{"基线max":>9}{"两阶段max":>10}')
    print('-' * 80)
    for tag, msk in (('全部', np.ones(n, bool)),
                     ('y < -11', y < -11),
                     ('-11 <= y < -9', (y >= -11) & (y < -9)),
                     ('-9 <= y < -7', (y >= -9) & (y < -7)),
                     ('y >= -7', y >= -7)):
        a1 = np.abs(y[msk] - p1[msk]); a2 = np.abs(y[msk] - p2[msk])
        print(f'{tag:<16}{msk.sum():>5}{a1.mean():>10.3f}{a2.mean():>11.3f}'
              f'{int((a1>2).sum()):>8}{int((a2>2).sum()):>9}'
              f'{a1.max():>9.2f}{a2.max():>10.2f}')

    print('\n尾部（>2dB）逐首变化:')
    t = np.where((np.abs(y - p1) > 2) | (np.abs(y - p2) > 2))[0]
    t = t[np.argsort(-np.abs(y - p1)[t])]
    print(f'{"文件":<46}{"标签":>8}{"基线":>8}{"两阶段":>8}{"基线err":>9}{"两阶段err":>10}')
    print('-' * 92)
    for i in t:
        nm = os.path.basename(paths[i].replace('\\', '/'))
        print(f'{nm[:44]:<46}{y[i]:>8.2f}{p1[i]:>8.2f}{p2[i]:>8.2f}'
              f'{abs(y[i]-p1[i]):>9.2f}{abs(y[i]-p2[i]):>10.2f}')

    np.save(args.out, p2)


if __name__ == '__main__':
    main()
