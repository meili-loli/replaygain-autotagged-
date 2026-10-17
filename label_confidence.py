# -*- coding: utf-8 -*-
"""标签可信度分级 + 分层的诚实指标。

为什么需要
  现在 >2dB=16 这个数字里混着 7 首「你自己两遍能差 1~2 dB」的歌。
  它们不是模型失败，是目标在动。混在一起会持续误导读数。

分级规则
  high    两轮都复核过，且两轮标的值完全一致        -> 可靠
  medium  只复核过一次，或两轮标的值不一致          -> 存疑
  low     从未人复核过（标签继承自旧模型/历史批次）   -> 无人工验证

输出
  label_confidence.csv   每首歌一行：标签、可信度、来源、复核轮次
  eval_by_confidence     分层指标（控制台 + CSV）
"""
import os
import sys
import json
import numpy as np
import joblib
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
TOL = 0.75


def load_candidates(path):
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path, encoding='utf-8-sig')
    return {os.path.basename(str(f)): (float(y), float(s))
            for f, y, s in zip(d['file'], d['y_label'], d['suggest'])}


def _load_oof_pred():
    """优先读 eval_out/per_idx.csv（train_eval_export.py 的产物）；
    找不到再退回 _oof_pred_10seed.npy（bench_* 脚本的产物）。"""
    per_idx = os.path.join(BASE, 'eval_out', 'per_idx.csv')
    if os.path.exists(per_idx):
        d = pd.read_csv(per_idx)
        return d.sort_values('idx')['pred'].to_numpy(np.float64), 'eval_out/per_idx.csv'
    npy = os.path.join(BASE, '_oof_pred_10seed.npy')
    if os.path.exists(npy):
        return np.load(npy), '_oof_pred_10seed.npy'
    return None, None


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='给标签打可信度分级（high/medium/low），并按分层重算指标。'
                    '需要先跑 train_eval_export.py 生成特征缓存与 eval_out/per_idx.csv。')
    ap.add_argument('--out', default=os.path.join(BASE, 'label_confidence.csv'))
    args = ap.parse_args()

    cache_p = os.path.join(BASE, 'handcraft_cache_42d_v2.pkl')
    if not os.path.exists(cache_p):
        sys.exit(f'找不到 {cache_p}\n请先跑一次 train_eval_export.py 生成特征缓存。')
    c = joblib.load(cache_p)
    y = np.asarray(c['y'], np.float64)
    paths = c['orig_paths']
    names = [os.path.basename(p.replace('\\', '/')) for p in paths]
    dirs = [os.path.dirname(p.replace('\\', '/')) for p in paths]

    pred, src = _load_oof_pred()
    if pred is None:
        sys.exit('找不到 OOF 预测。请先跑 train_eval_export.py（生成 eval_out/per_idx.csv）。')
    if len(pred) != len(y):
        sys.exit(f'OOF 预测长度 {len(pred)} 与缓存样本数 {len(y)} 不一致，'
                 f'请重新跑一次 train_eval_export.py。')
    print(f'OOF 预测来源: {src}')

    prog_p = os.path.join(BASE, 'annotate_progress_review.json')
    if not os.path.exists(prog_p):
        print(f'[WARN] 找不到 {os.path.basename(prog_p)}，'
              f'所有标签将标记为 low（从未人工复核）。')
        prog = {'annotations': {}}
    else:
        prog = json.load(open(prog_p, encoding='utf-8'))
    final = {os.path.basename(p.replace('\\', '/')): float(v)
             for p, v in prog['annotations'].items()}
    r1 = load_candidates(os.path.join(BASE, 'review_candidates_round1.csv'))
    r2 = load_candidates(os.path.join(BASE, 'review_candidates_round2_ordered.csv'))

    rows = []
    for n, d, yy, pp in zip(names, dirs, y, pred):
        in1, in2 = n in r1, n in r2
        # 第二轮开始时的值：r2 的 y_label（第二轮候选清单里的原始值）
        if in2:
            val_after_r1 = r2[n][0]
        elif in1:
            val_after_r1 = final.get(n)
        else:
            val_after_r1 = None
        cur = final.get(n)

        if in1 and in2:
            if abs(val_after_r1 - cur) < 1e-9:
                conf, note = 'high', '两轮复核一致'
            else:
                conf, note = 'medium', f'两轮复核不一致 {val_after_r1:+.2f}->{cur:+.2f}'
        elif in1 or in2:
            # 只复核一次：判断是否按 A 采用了建议（等于 suggest 说明可能半自动）
            cand = r2.get(n) or r1.get(n)
            sug = cand[1] if cand else None
            used_sug = sug is not None and cur is not None and abs(cur - sug) < 1e-9
            conf = 'medium'
            note = ('复核一次，值等于建议值(疑按A)' if used_sug
                    else '复核一次，人工确认值')
        else:
            conf, note = 'low', '从未人工复核'

        rows.append(dict(file=n, dir=d, y=yy, pred=pp, ae=abs(yy - pp),
                         resid=yy - pp, confidence=conf, note=note,
                         rounds=int(in1) + int(in2)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, 'label_confidence.csv'), index=False,
              encoding='utf-8-sig')

    print('=' * 100)
    print(f'标签可信度分级（{len(df)} 首）')
    print('=' * 100)
    cnt = df.groupby('confidence').size()
    for k in ('high', 'medium', 'low'):
        print(f'  {k:<8} {int(cnt.get(k, 0)):>4} 首  ({100*cnt.get(k,0)/len(df):5.1f}%)')
    print()
    for k in ('high', 'medium', 'low'):
        sub = df[df.confidence == k]
        if sub.empty:
            continue
        print(f'  [{k}] 来源明细:')
        for note, g in sub.groupby('note'):
            print(f'      {note:<34} {len(g):>4} 首')

    print()
    print('=' * 100)
    print('分层指标（这才是模型的真实能力读数）')
    print('=' * 100)
    hdr = (f'{"分层":<22}{"n":>5}{"MAE":>9}{"<=0.75":>9}{">1dB":>7}'
           f'{">2dB":>7}{">3dB":>7}{"max":>8}')
    print(hdr)
    print('-' * 100)
    out = []
    for k in ('high', 'medium', 'low', 'ALL'):
        sub = df if k == 'ALL' else df[df.confidence == k]
        if sub.empty:
            continue
        a = sub.ae.values
        rec = dict(conf=k, n=len(sub), mae=a.mean(),
                   le075=float((a <= TOL).mean()), gt1=int((a > 1).sum()),
                   gt2=int((a > 2).sum()), gt3=int((a > 3).sum()), mx=a.max())
        out.append(rec)
        print(f'{k:<22}{rec["n"]:>5}{rec["mae"]:>9.4f}{100*rec["le075"]:>8.1f}%'
              f'{rec["gt1"]:>7}{rec["gt2"]:>7}{rec["gt3"]:>7}{rec["mx"]:>8.2f}')
    # 合并 high+medium = 人工参与过的
    hm = df[df.confidence.isin(['high', 'medium'])]
    if len(hm):
        a = hm.ae.values
        print(f'{"high+medium（人复核过）":<22}{len(hm):>5}{a.mean():>9.4f}'
              f'{100*(a<=TOL).mean():>8.1f}%{int((a>1).sum()):>7}'
              f'{int((a>2).sum()):>7}{int((a>3).sum()):>7}{a.max():>8.2f}')
    pd.DataFrame(out).to_csv(os.path.join(BASE, 'eval_by_confidence.csv'),
                            index=False, encoding='utf-8-sig')

    print()
    print('=' * 100)
    print('尾部（>2dB）16 首按可信度归类')
    print('=' * 100)
    t = df[df.ae > 2].sort_values('ae', ascending=False)
    for _, r in t.iterrows():
        print(f'  {r["file"][:44]:<46}{r.y:>8.2f}{r.pred:>8.2f}{r.ae:>7.2f}  '
              f'[{r.confidence}] {r.note}')
    print()
    print('剔除「标签在动」(medium) 后的尾部:', int((df[df.confidence != "medium"].ae > 2).sum()),
          '首')
    print('-> label_confidence.csv / eval_by_confidence.csv')


if __name__ == '__main__':
    main()
