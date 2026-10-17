# -*- coding: utf-8 -*-
"""
诚实评估 —— 干净协议下的模型诊断。

要点：
  1. GroupKFold：同一首歌的"原始样本"和"增强样本"锁在同一折，杜绝 OOF 泄漏
  2. 只在 909 首原始样本上计分（增强样本只是训练时的辅助，不作为评估对象）
  3. 单模型评估（生产环境就是单模型，5 模型平均的读数没意义）

跑法：  python diag_honest_eval.py
输出：  控制台表格 + diag_honest_eval.csv
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CACHE = 'handcraft_cache_42d_aug05.pkl'   # 含增强样本的旧缓存，用于演示泄漏
# 注意：这是「泄漏量化」的演示脚本。它故意使用旧的「原始+增强堆叠」缓存来复现
# 朴素 KFold 会把同一首歌的增强副本分到不同折的问题。
# 若手上没有这个缓存，脚本会提示；正常训练请用 train_eval_export.py。
N_ORIG = 909                 # 原始样本数（前 N_ORIG 行）
LOW_THR, A_THR = -10.0, -9.0  # shift=-5 后的专家训练阈值
TAU = 0.77                    # meta.json 里的门控阈值
SEED_REPS = [0, 1, 2]         # 重复几次不同的折划分


# ------------------------------------------------------------------
# 模型构造（与 train_eval_export.py 保持一致）
# ------------------------------------------------------------------
def train_reg(Xtr, ytr, seed=0):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          min_child_samples=10, random_state=seed,
                          verbose=-1, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m


def train_gate(G, t, seed=0):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                           min_child_samples=10, random_state=seed, verbose=-1,
                           n_jobs=-1, scale_pos_weight=16.0)
    m.fit(G, t)
    return m


def f6(lp, ap):
    d = ap - lp
    return np.column_stack([lp, ap, d, np.abs(d),
                            np.minimum(lp, ap), np.maximum(lp, ap)]).astype(np.float32)


def dual_expert_predict(X_tr, y_tr, X_va, soft_delta=0.0):
    """复刻 train_eval_export.py 的双专家 + 二分类门控。"""
    lm_ = y_tr < LOW_THR
    am_ = y_tr >= A_THR
    if lm_.sum() < 30:
        lm_ = np.ones(len(y_tr), bool)
    if am_.sum() < 10:
        am_ = np.ones(len(y_tr), bool)

    low_oof = np.zeros(len(y_tr), np.float32)
    a_oof = np.zeros(len(y_tr), np.float32)
    for it, iv in KFold(5, shuffle=True, random_state=777).split(X_tr):
        l2 = y_tr[it] < LOW_THR
        a2 = y_tr[it] >= A_THR
        if l2.sum() < 10:
            l2 = np.ones(len(y_tr[it]), bool)
        if a2.sum() < 5:
            a2 = np.ones(len(y_tr[it]), bool)
        low_oof[iv] = train_reg(X_tr[it][l2], y_tr[it][l2]).predict(X_tr[iv])
        a_oof[iv] = train_reg(X_tr[it][a2], y_tr[it][a2]).predict(X_tr[iv])

    G_tr = np.hstack([X_tr, f6(low_oof, a_oof)])
    y_bin = ((np.abs(y_tr - low_oof) - np.abs(y_tr - a_oof)) > 0).astype(int)

    lp = train_reg(X_tr[lm_], y_tr[lm_]).predict(X_va)
    ap = train_reg(X_tr[am_], y_tr[am_]).predict(X_va)
    pr = train_gate(G_tr, y_bin).predict_proba(
        np.hstack([X_va, f6(lp, ap)]))[:, 1]

    hard = np.where(pr > TAU, ap, lp)
    w = np.clip((pr - (TAU - soft_delta)) / (2 * max(soft_delta, 1e-9)), 0, 1)
    soft = lp + w * (ap - lp)
    return hard, soft


# ------------------------------------------------------------------
# 实验
# ------------------------------------------------------------------
def load():
    c = joblib.load(CACHE)
    X = np.asarray(c['X'], np.float32)
    y = np.asarray(c['y'], np.float32)
    return X, y, X[:N_ORIG], y[:N_ORIG]


def leaderboard(X, y, Xo, yo):
    """干净协议下的方案排行榜。"""
    groups = np.tile(np.arange(N_ORIG), 2)   # 原始与增强共享 group id
    acc = {k: [] for k in ['单LGB', 'Ridge42', '双专家_硬路由', '双专家_软路由0.2']}

    for sd in SEED_REPS:
        for tr, va in GroupKFold(5).split(X, y, groups):
            mo = va < N_ORIG          # 只在原始样本上计分
            if mo.sum() == 0:
                continue
            y_va = y[va][mo]
            X_va = X[va][mo]

            acc['单LGB'].append(np.abs(y_va - train_reg(X[tr], y[tr], sd).predict(X_va)))

            sc = StandardScaler().fit(X[tr])
            rid = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
            acc['Ridge42'].append(np.abs(y_va - rid.predict(sc.transform(X_va))))

            hard, soft = dual_expert_predict(X[tr], y[tr], X_va, soft_delta=0.2)
            acc['双专家_硬路由'].append(np.abs(y_va - hard))
            acc['双专家_软路由0.2'].append(np.abs(y_va - soft))

    return {k: float(np.concatenate(v).mean()) for k, v in acc.items()}


def leakage_check(X, y):
    """量化增强样本泄漏：朴素 KFold vs GroupKFold。"""
    groups = np.tile(np.arange(N_ORIG), 2)
    out = {}
    for name, splitter in [('朴素KFold(有泄漏)', 'naive'), ('GroupKFold(无泄漏)', 'group')]:
        errs = []
        for sd in SEED_REPS:
            if splitter == 'naive':
                it = KFold(5, shuffle=True, random_state=sd).split(X)
            else:
                it = GroupKFold(5).split(X, y, groups)
            for tr, va in it:
                mo = va < N_ORIG
                if mo.sum() == 0:
                    continue
                hard, _ = dual_expert_predict(X[tr], y[tr], X[va][mo])
                errs.append(np.abs(y[va][mo] - hard))
        out[name] = float(np.concatenate(errs).mean())
    return out


def learning_curve(Xo, yo):
    """单 LGB 的学习曲线（纯留出，无增强）。"""
    rows = []
    for n in [200, 300, 450, 600, 750]:
        rng = np.random.RandomState(11)
        ms = []
        for rep in range(4):
            idx = rng.permutation(N_ORIG)
            tr, te = idx[:n], idx[n:]
            m = train_reg(Xo[tr], yo[tr], rep)
            ms.append(np.abs(yo[te] - m.predict(Xo[te])).mean())
        rows.append((n, float(np.mean(ms))))
    return rows


def augmentation_check(X, y, Xo, yo):
    """增强样本有没有用（干净协议）。"""
    groups = np.tile(np.arange(N_ORIG), 2)
    res = {}
    for use_aug in (False, True):
        errs = []
        for sd in SEED_REPS:
            for tr, va in GroupKFold(5).split(X, y, groups):
                mo = va < N_ORIG
                if mo.sum() == 0:
                    continue
                if use_aug:
                    tr_ = tr
                    Xtr, ytr = X[tr_], y[tr_]
                else:
                    tr_ = tr[tr < N_ORIG]
                    Xtr, ytr = Xo[tr_], yo[tr_]
                errs.append(np.abs(y[va][mo] - train_reg(Xtr, ytr, sd).predict(X[va][mo])))
        res['含增强' if use_aug else '只用原始'] = float(np.concatenate(errs).mean())
    return res


def main():
    t0 = time.time()
    X, y, Xo, yo = load()
    print(f'X={X.shape}  y={y.shape}  原始样本={N_ORIG}\n')

    print('=' * 62)
    print('【1】泄漏量化  (双专家+门控, 硬路由)')
    print('=' * 62)
    lk = leakage_check(X, y)
    for k, v in lk.items():
        print(f'  {k:22s}  MAE = {v:.4f}')
    print(f'  -> 泄漏量 = {lk["GroupKFold(无泄漏)"] - lk["朴素KFold(有泄漏)"]:.4f} dB\n')

    print('=' * 62)
    print('【2】干净协议方案排行榜  (GroupKFold, 只在原始样本计分)')
    print('=' * 62)
    lb = leaderboard(X, y, Xo, yo)
    for k, v in sorted(lb.items(), key=lambda x: x[1]):
        print(f'  {k:22s}  MAE = {v:.4f}')

    print()
    print('=' * 62)
    print('【3】增强样本有没有用')
    print('=' * 62)
    for k, v in augmentation_check(X, y, Xo, yo).items():
        print(f'  {k:22s}  MAE = {v:.4f}')

    print()
    print('=' * 62)
    print('【4】单 LGB 学习曲线  (纯留出, 无增强)')
    print('=' * 62)
    rows = learning_curve(Xo, yo)
    for n, m in rows:
        print(f'  训练 {n:4d} 首            MAE = {m:.4f}')
    ns = np.array([r[0] for r in rows], float)
    ms = np.array([r[1] for r in rows], float)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(ns), 1 / np.sqrt(ns)]),
                               ms, rcond=None)
    print(f'\n  拟合 MAE = {coef[0]:.4f} + {coef[1]:.2f}/sqrt(n)')
    print(f'  外推 1800 首: {coef[0] + coef[1] / np.sqrt(1800):.4f}'
          f'   外推 3600 首: {coef[0] + coef[1] / np.sqrt(3600):.4f}'
          f'   渐近下限: {coef[0]:.4f}')

    out = []
    out += [{'实验': '泄漏量化', '配置': k, 'MAE': round(v, 4)} for k, v in lk.items()]
    out += [{'实验': '方案排行榜', '配置': k, 'MAE': round(v, 4)} for k, v in lb.items()]
    out += [{'实验': '增强有效性', '配置': k, 'MAE': round(v, 4)}
            for k, v in augmentation_check(X, y, Xo, yo).items()]
    out += [{'实验': '学习曲线', '配置': f'{n}首', 'MAE': round(m, 4)} for n, m in rows]
    pd.DataFrame(out).to_csv('diag_honest_eval.csv', index=False, encoding='utf-8-sig')
    print(f'\n用时 {time.time() - t0:.0f}s  ->  diag_honest_eval.csv')


if __name__ == '__main__':
    main()
