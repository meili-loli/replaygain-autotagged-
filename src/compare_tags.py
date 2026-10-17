# -*- coding: utf-8 -*-
"""
对比「新模型预测」和「文件里现有的 ReplayGain 标签」。

用来判断要不要重扫：如果绝大多数歌只差零点几 dB，重扫没意义；
如果有一批差 2 dB 以上，那批就是重扫的收益。

输入
  inference_logs/predict_*.csv    predict_and_tag.py --dry-run 的输出
输出
  tag_compare.csv   逐条对比
  控制台摘要

用法
  python compare_tags.py                          # 用最新的推理日志
  python compare_tags.py --log inference_logs/xxx.csv
  python compare_tags.py --min-diff 1.0           # 只看差 >= 1dB 的
"""

import os
import re
import sys
import io
import glob
import argparse

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import mutagen
    from mutagen.flac import FLAC
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, ID3NoHeaderError
    from mutagen.mp4 import MP4
except ImportError:
    sys.exit('请先 pip install mutagen')

GAIN_RE = re.compile(r'([-+]?\d+(?:\.\d+)?)\s*dB')


def read_current_gain(path):
    """读文件里现有的 ReplayGain track gain，返回 float 或 None。"""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.flac':
            a = FLAC(path)
            for k in a:
                if k.lower() == 'replaygain_track_gain':
                    v = a[k][0] if isinstance(a[k], list) else a[k]
                    m = GAIN_RE.search(str(v))
                    return float(m.group(1)) if m else None
            return None
        if ext == '.mp3':
            try:
                t = ID3(path)
            except ID3NoHeaderError:
                return None
            for k in t:
                if k.upper().startswith('TXXX:REPLAYGAIN_TRACK_GAIN'):
                    m = GAIN_RE.search(str(t[k].text[0]))
                    return float(m.group(1)) if m else None
            return None
        if ext in ('.m4a', '.mp4', '.m4b', '.m4r'):
            a = MP4(path)
            if a.tags is None:
                return None
            v = a.tags.get('----:com.apple.iTunes:REPLAYGAIN_TRACK_GAIN')
            if not v:
                return None
            m = GAIN_RE.search(v[0].decode('utf-8', 'ignore'))
            return float(m.group(1)) if m else None
        a = mutagen.File(path, easy=True)
        if a is None or a.tags is None:
            return None
        for k in a.tags:
            if k.lower() == 'replaygain_track_gain':
                v = a.tags[k]
                m = GAIN_RE.search(str(v[0] if isinstance(v, list) else v))
                return float(m.group(1)) if m else None
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=None)
    ap.add_argument('--out', default='tag_compare.csv')
    ap.add_argument('--min-diff', type=float, default=0.0)
    args = ap.parse_args()

    log = args.log
    if log is None:
        cands = sorted(glob.glob(os.path.join('inference_logs', 'predict_*.csv')),
                       key=os.path.getmtime)
        if not cands:
            sys.exit('inference_logs/ 里没有日志')
        log = cands[-1]
    print(f'读取 {log}')

    d = pd.read_csv(log, encoding='utf-8-sig')
    d = d[d['y_pred'].notna()]
    print(f'  日志里 {len(d)} 条预测')

    rows = []
    n_notag = 0
    for _, r in d.iterrows():
        p = r['path']
        cur = read_current_gain(p)
        if cur is None:
            n_notag += 1
            continue
        pred = float(r['y_pred'])
        rows.append({
            'file': os.path.basename(p),
            'dir': os.path.dirname(p),
            'path': p,
            'tag_now': cur,
            'new_pred': round(pred, 2),
            'diff': round(pred - cur, 2),
            'abs_diff': round(abs(pred - cur), 2),
            'spread': r.get('spread', ''),

        })

    df = pd.DataFrame(rows)
    if df.empty:
        print('没有任何文件带 ReplayGain 标签')
        return

    print(f'  有标签 {len(df)} 条   无标签 {n_notag} 条')
    print()
    a = df['abs_diff']
    print('=' * 62)
    print(f'新模型预测 vs 现有标签   （{len(df)} 首）')
    print('=' * 62)
    print(f'  MAE          = {a.mean():.3f} dB')
    print(f'  中位数       = {a.median():.3f} dB')
    print(f'  p90 / p95    = {a.quantile(0.9):.2f} / {a.quantile(0.95):.2f} dB')
    print(f'  max          = {a.max():.2f} dB')
    print()
    for t in [0.5, 1.0, 1.5, 2.0, 3.0]:
        n = int((a > t).sum())
        print(f'  差 > {t:.1f} dB : {n:5d} 首  ({100 * n / len(df):5.1f}%)')
    print()

    out = df.sort_values('abs_diff', ascending=False)
    if args.min_diff > 0:
        out = out[out.abs_diff >= args.min_diff]
    out.drop(columns=['abs_diff']).to_csv(args.out, index=False,
                                          encoding='utf-8-sig')
    print(f'明细 -> {args.out}   ({len(out)} 条)')
    if args.min_diff <= 0:
        print()
        print('差异最大的 25 首:')
        print(f'{"文件":<48}{"现在":>8}{"新模型":>9}{"差":>8}')
        print('-' * 73)
        for _, r in out.head(25).iterrows():
            print(f'{r["file"][:46]:<48}{r.tag_now:>+8.2f}{r.new_pred:>+9.2f}'
                  f'{r["diff"]:>+8.2f}')


if __name__ == '__main__':
    main()
