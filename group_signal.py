# -*- coding: utf-8 -*-
"""组信号特征（目录 / 专辑 / 艺人）—— 训练与推理共用的唯一实现。

为什么需要这个文件
  组信号在训练和推理时都必须用**完全相同**的定义。分别实现是经典 bug 来源，
  所以这里只写一份，两边 import。

生产流程（两阶段）
  Phase 1  一阶段模型（42 维）对**全库**预测 -> base_pred
  Phase 2  按分组键算「组内其他成员 base_pred 均值 + 其余成员数」
  Phase 3  二阶段模型用 [42 维 + 组信号] 出最终预测

关键约束
  计算某首的组信号时，必须排除它自己的 base_pred —— 否则等于把答案喂回去。
"""
import os
import numpy as np
import pandas as pd

META_PATH = 'library_meta.csv'
GROUP_KEYS = ('dir', 'album', 'artist')
DEFAULT_KEYS = ('dir', 'album', 'artist')
N_GROUP_FEATS = {1: 2, 2: 4, 3: 6}


def norm_path(p):
    """跨格式统一的路径 key（与 build_library_meta.py 中同名函数一致）。"""
    q = str(p).replace('\\', '/')
    while '//' in q:
        q = q.replace('//', '/')
    if len(q) > 1 and q[1] == ':':
        q = q[0].lower() + q[1:]
    return q


def load_meta(path=META_PATH):
    """读 library_meta.csv -> {norm_path: {'dir','album','artist'}}"""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, encoding='utf-8-sig')
    out = {}
    for _, r in df.iterrows():
        key = r['path_key'] if 'path_key' in df.columns else norm_path(r['path'])
        out[key] = {'dir': r['dir_key'], 'album': r['album'],
                    'artist': r['artist']}
    return out


def _fallback_keys(path):
    """元数据缺失时的兜底：用路径和文件名推。"""
    d = norm_path(os.path.dirname(os.path.abspath(path)))
    name = os.path.basename(os.path.dirname(path)) or d
    return {'dir': d, 'album': name, 'artist': name}


def keys_for(paths, meta=None, use=GROUP_KEYS):
    """返回 {key_name: list[str]}，长度与 paths 一致。"""
    if meta is None:
        meta = load_meta()
    out = {k: [] for k in use}
    for p in paths:
        info = meta.get(norm_path(p)) or _fallback_keys(p)
        for k in use:
            out[k].append(str(info.get(k, '') or ''))
    return out


def group_signal(keys, values, self_idx=None):
    """组内「其他成员」的均值 + 其余成员数。

    keys     与 values 等长的分组键
    values   基准预测（通常是 42 维一阶段模型的输出）
    self_idx 要计算信号的行号；None 表示全部行
    返回 (mean[n], cnt[n])；单曲组的 mean 填 -9.0（哨兵）、cnt 填 0
    """
    values = np.asarray(values, np.float64)
    n = len(values)
    mean = np.full(n, -9.0)
    cnt = np.zeros(n)
    rows = range(n) if self_idx is None else np.asarray(self_idx)
    s = pd.Series(keys)
    # 一次性建好 group -> 行号 的映射
    groups = {}
    for g, idx in s.groupby(s, sort=False).groups.items():
        groups[g] = np.asarray(list(idx))
    for i in rows:
        idx = groups.get(keys[i])
        if idx is None or len(idx) <= 1:
            continue
        tot = values[idx].sum()
        mean[i] = (tot - values[i]) / (len(idx) - 1)
        cnt[i] = len(idx) - 1
    return mean, cnt


def build_group_features(paths, base_pred, meta=None, use=DEFAULT_KEYS):
    """按 use 指定的分组键拼出组信号矩阵，形状 (n, 2*len(use))。"""
    ks = keys_for(paths, meta, use)
    cols = []
    for k in use:
        m, c = group_signal(ks[k], base_pred)
        cols.append(m)
        cols.append(c)
    if not cols:
        return np.zeros((len(paths), 0)), use
    return np.column_stack(cols), use


def n_features_for(use):
    return N_GROUP_FEATS.get(len(list(use)), 2 * len(list(use)))
