# -*- coding: utf-8 -*-
"""模型导出工具：模型卡生成 + 原生格式转换 + 发布打包（不改动原模型目录）。

把 train_eval_export.py 训练出的模型目录（.pkl）安全地导出为可分享的
模型发布包（zip）：

    python model_export.py pack --model-dir model_dir \\
        --csv delta_gain.csv --audio-root D:\\Music \\
        --author 你的署名 --note 一句话介绍 [--publish-tracks] \\
        --out rg-model-xxx.zip

做三件事：
  1. convert  pickle 模型 → LightGBM 原生 .txt（加载时不反序列化代码，
              这是模型能安全分发的前提）。只在导出目录里生成，原模型不动
  2. card     结合 meta.json + delta_gain.csv + 音频标签生成 MODEL_CARD.json
              （样本数 / MAE / 口味摘要；歌单默认隐藏，--publish-tracks 才公开）
  3. pack     以上 + meta.json 打成发布 zip，打印 sha256

安全设计：发布包里只有 .txt / meta.json / MODEL_CARD.json 三类文件，
加载端（predict_and_tag.py）只走 LightGBM 原生解析器，绝不 joblib.load。
"""
import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime

AUDIO_EXTS = {'.flac', '.mp3', '.m4a', '.mp4', '.m4b', '.m4r',
              '.ogg', '.opus', '.wav', '.aiff', '.aif', '.ape', '.wv'}
PACK_FORMAT_VERSION = 3


# ---------------------------------------------------------------- 特征清单
def read_feature_order(scripts_dir):
    """从 predict_and_tag.py 用 ast 安全解析 FEATURE_ORDER（42 维名单）。"""
    src = os.path.join(scripts_dir, 'predict_and_tag.py')
    with open(src, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, 'id', None) == 'FEATURE_ORDER':
                    return list(ast.literal_eval(node.value))
    raise SystemExit('没在 predict_and_tag.py 里找到 FEATURE_ORDER 喵')


def read_cache_version(scripts_dir):
    """从 train_eval_export.py 解析 CACHE_VERSION（本地特征管线版本）。"""
    src = os.path.join(scripts_dir, 'train_eval_export.py')
    with open(src, encoding='utf-8') as f:
        import re
        m = re.search(r'CACHE_VERSION\s*=\s*(\d+)', f.read())
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------- 模型转换
def convert_models(model_dir, out_dir):
    """.pkl → 原生格式（原目录不动）；已是 .txt 的直接复制。返回文件清单。

    用 dump_model() + 纯 Python 写文件，而不是 booster.save_model()：
    LightGBM 的 C++ 层在 Windows 上写不了含非 ASCII 字符的路径
    （比如 C:\\Users\\某人\\...），dump 出 JSON 自己写就没有这个限制，
    加载端 lgb.Booster(model_str=...) 两种格式都认。
    """
    import joblib
    import lightgbm as lgb

    saved = []
    for dirpath, dirnames, filenames in os.walk(model_dir):
        dirnames.sort()
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, model_dir).replace('\\', '/')
            base, ext = os.path.splitext(fn)
            if ext == '.txt':
                dst = os.path.join(out_dir, rel)
                os.makedirs(os.path.dirname(dst) or out_dir, exist_ok=True)
                shutil.copyfile(full, dst)
                saved.append(rel)
            elif ext == '.pkl' and (base == 'model' or base.startswith('model_')):
                m = joblib.load(full)  # 本地自己的模型，可信场景
                booster = m.booster_ if hasattr(m, 'booster_') else m
                # model_to_string() = save_model 的内容但不经 C++ 写文件，
                # 绕开 LightGBM 在 Windows 上写不了非 ASCII 路径的限制
                model_text = booster.model_to_string()
                dst = os.path.join(out_dir, rel.replace('.pkl', '.txt'))
                os.makedirs(os.path.dirname(dst) or out_dir, exist_ok=True)
                with open(dst, 'w', encoding='utf-8') as f:
                    f.write(model_text)
                saved.append(rel.replace('.pkl', '.txt'))
    if not saved:
        raise SystemExit(f'在 {model_dir} 里没找到可导出的模型文件（model*.pkl/.txt）喵')
    return saved


# ---------------------------------------------------------------- 模型卡
def _norm(p):
    k = str(p).replace('\\', '/')
    while '//' in k:
        k = k.replace('//', '/')
    return k.lower()


def read_tags(path):
    try:
        import mutagen
        m = mutagen.File(path, easy=True)
        if m is None:
            return '', '', ''
        def g(key):
            v = m.get(key)
            if isinstance(v, list):
                return v[0] if v else ''
            return v if isinstance(v, str) else ''
        return g('title'), g('artist'), g('album')
    except Exception:
        return '', '', ''


def read_marks(csv_path):
    """delta_gain.csv → {basename: gain}（与训练端同为 basename 匹配）。"""
    import csv
    marks = {}
    if not os.path.exists(csv_path):
        return marks
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            fn = (r.get('file') or r.get('filename') or r.get('path') or '').strip()
            g = (r.get('delta_gain') or '').strip()
            if fn and g:
                try:
                    marks[os.path.basename(fn)] = float(g)
                except ValueError:
                    pass
    return marks


def build_card(model_dir, meta, scripts_dir, csv_path, audio_root,
               author, note, publish_tracks):
    import numpy as np
    fo = read_feature_order(scripts_dir)
    y_marks = read_marks(csv_path)

    tracks = []
    if publish_tracks and audio_root and os.path.isdir(audio_root):
        for dirpath, dirnames, filenames in os.walk(audio_root):
            dirnames.sort()
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() not in AUDIO_EXTS:
                    continue
                if fn not in y_marks:
                    continue
                full = os.path.join(dirpath, fn)
                title, artist, album = read_tags(full)
                tracks.append({'file': fn, 'title': title, 'artist': artist,
                               'album': album, 'gain': round(y_marks[fn], 2)})
    tracks.sort(key=lambda t: t['file'])

    gains = np.array(sorted(y_marks.values()), dtype=float)
    st = meta.get('oof_stats') or {}
    return {
        'card_version': 1,
        'model_name': f"rg-model-{author or 'anonymous'}-{datetime.now().strftime('%Y%m%d')}",
        'author': author or 'anonymous',
        'note': note or '',
        'model_type': meta.get('model_type'),
        'format_version': PACK_FORMAT_VERSION,
        'feature_pipeline_version': read_cache_version(scripts_dir),
        'feature_order': fo,
        'lightgbm': meta.get('lightgbm'),
        'python': meta.get('python'),
        'n_ensemble': meta.get('n_ensemble'),
        'label_shift': meta.get('label_shift'),
        'sample_count': int(meta.get('n_train_samples') or len(y_marks)),
        'oof_stats': st,
        'taste': {
            'gain_mean': round(float(gains.mean()), 3) if len(gains) else None,
            'gain_p25': round(float(np.percentile(gains, 25)), 3) if len(gains) else None,
            'gain_p50': round(float(np.percentile(gains, 50)), 3) if len(gains) else None,
            'gain_p75': round(float(np.percentile(gains, 75)), 3) if len(gains) else None,
        },
        'tracks_public': bool(publish_tracks),
        'tracks': tracks,
        'trained_at': meta.get('trained_at'),
        'exported_at': datetime.now().isoformat(timespec='seconds'),
    }


# ---------------------------------------------------------------- 打包
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def do_pack(args):
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(args.model_dir, 'meta.json')
    if not os.path.isfile(meta_path):
        raise SystemExit(f'{args.model_dir} 里没有 meta.json 喵（先用 train_eval_export.py 训练）')
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)

    work = args.workdir or tempfile.mkdtemp(prefix='rg-export-')
    os.makedirs(work, exist_ok=True)
    print(f'[1/3] 转换模型为原生格式 -> {work}')
    saved = convert_models(args.model_dir, work)

    print('[2/3] 生成模型卡')
    card = build_card(args.model_dir, meta, scripts_dir, args.csv,
                      args.audio_root, args.author, args.note,
                      args.publish_tracks)
    with open(os.path.join(work, 'MODEL_CARD.json'), 'w', encoding='utf-8') as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    meta_out = dict(meta)
    meta_out.update({'format_version': PACK_FORMAT_VERSION,
                     'feature_pipeline_version': card['feature_pipeline_version'],
                     'feature_order': card['feature_order']})
    with open(os.path.join(work, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    print('[3/3] 打包发布 zip')
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(work):
            dirnames.sort()
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                z.write(full, os.path.relpath(full, work).replace('\\', '/'))
    digest = sha256_file(args.out)
    st = card['oof_stats'] or {}
    print(f'发布包已生成：{args.out}')
    print(f'  模型文件 {len(saved)} 个  样本 {card["sample_count"]} 首  '
          f'MAE {st.get("mae", "?")}  歌单{"公开" if card["tracks_public"] else "隐藏"}')
    print(f'  sha256 = {digest}')
    shutil.rmtree(work, ignore_errors=True) if not args.keep_workdir else print(f'  中间目录保留在 {work}')


def do_convert(args):
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    read_feature_order(scripts_dir)  # 确认脚本齐
    saved = convert_models(args.model_dir, args.out)
    print(f'已转换 {len(saved)} 个模型文件 -> {args.out}')
    for s in saved:
        print('  ', s)


def do_card(args):
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(args.model_dir, 'meta.json')
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    card = build_card(args.model_dir, meta, scripts_dir, args.csv,
                      args.audio_root, args.author, args.note,
                      args.publish_tracks)
    out = args.out or os.path.join(args.model_dir, 'MODEL_CARD.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f'模型卡已写入 {out}')


def main():
    ap = argparse.ArgumentParser(description='模型导出工具（模型卡 / 原生格式 / 发布包）')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_pack = sub.add_parser('pack', help='一条龙：转换 + 模型卡 + 发布 zip')
    p_pack.add_argument('--model-dir', required=True)
    p_pack.add_argument('--csv', default='delta_gain.csv',
                        help='标注 CSV（默认 delta_gain.csv，模型卡用它统计口味/歌单）')
    p_pack.add_argument('--audio-root', default='',
                        help='音乐目录（--publish-tracks 时用于读标签）')
    p_pack.add_argument('--author', default='')
    p_pack.add_argument('--note', default='')
    p_pack.add_argument('--publish-tracks', action='store_true')
    p_pack.add_argument('--out', required=True)
    p_pack.add_argument('--workdir', default='', help='中间目录（默认临时目录）')
    p_pack.add_argument('--keep-workdir', action='store_true')

    p_conv = sub.add_parser('convert', help='仅转换模型为原生 .txt')
    p_conv.add_argument('--model-dir', required=True)
    p_conv.add_argument('--out', required=True)

    p_card = sub.add_parser('card', help='仅生成 MODEL_CARD.json')
    p_card.add_argument('--model-dir', required=True)
    p_card.add_argument('--csv', default='delta_gain.csv')
    p_card.add_argument('--audio-root', default='')
    p_card.add_argument('--author', default='')
    p_card.add_argument('--note', default='')
    p_card.add_argument('--publish-tracks', action='store_true')
    p_card.add_argument('--out', default='')

    args = ap.parse_args()
    if args.cmd == 'pack':
        do_pack(args)
    elif args.cmd == 'convert':
        do_convert(args)
    elif args.cmd == 'card':
        do_card(args)


if __name__ == '__main__':
    main()
