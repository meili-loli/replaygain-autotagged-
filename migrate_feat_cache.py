# -*- coding: utf-8 -*-
r"""
feat_cache 路径迁移。

背景
----
feat_cache/<md5(绝对路径)>.npz 的 key 是绝对路径的 md5。
音频库从 C:\Users\xieya\Music\我的音乐\  挪到  E:\我的音乐\，
而且中间还重新分过文件夹（比如 7yue\10 Lovers.flac 变成 32\Music\10 Lovers.flac），
所以所有 key 都对不上，会触发全量重提（约 80 分钟）。

本脚本按【文件名】在音频库里重新定位每首歌，然后把旧 key 重命名为新 key。
匹配规则与 train_eval_export.py 的 match_csv_to_files 保持一致
（NFC 归一化 + 特殊字符替换 + 全小写）。

用法
----
  python migrate_feat_cache.py --dry-run            # 只看不动
  python migrate_feat_cache.py                      # 实际迁移
  python migrate_feat_cache.py --new-root "E:\我的音乐"
"""

import os
import sys
import io
import shutil
import hashlib
import argparse
import unicodedata

import joblib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

FEAT_CACHE = 'feat_cache'
OLD_LISTING = 'handcraft_cache_42d_aug05.pkl'   # 含 orig_paths 的旧缓存

AUDIO_EXTS = {'.flac', '.wav', '.mp3', '.m4a', '.mp4', '.m4b', '.m4r',
              '.ogg', '.opus', '.aiff', '.aif', '.ape', '.wv'}


def key_of(path):
    return hashlib.md5(os.path.abspath(path).encode('utf-8')).hexdigest()


def normalize_filename(s):
    """与 train_eval_export.py 完全一致。"""
    s = unicodedata.normalize('NFC', str(s))
    for old, new in {'Ø': 'O', 'ø': 'o', 'Æ': 'AE', 'æ': 'ae', 'ß': 'ss',
                     'Ł': 'L', 'ł': 'l', 'Œ': 'OE', 'œ': 'oe'}.items():
        s = s.replace(old, new)
    return s.lower().strip()


def build_index(root):
    idx = {}
    dup = 0
    for dp, _, fns in os.walk(root):
        for f in fns:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                k = normalize_filename(f)
                if k in idx:
                    dup += 1
                else:
                    idx[k] = os.path.join(dp, f)
    return idx, dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new-root',
                    default=os.path.join('E:', os.sep, '我的音乐'))
    ap.add_argument('--listing', default=OLD_LISTING)
    ap.add_argument('--cache-dir', default=FEAT_CACHE)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    print(f'new-root = {args.new_root}')
    print(f'扫描音频库 ...', flush=True)
    idx, n_dup = build_index(args.new_root)
    print(f'  索引到 {len(idx)} 个文件（同名冲突 {n_dup} 个）')

    c = joblib.load(args.listing)
    old_paths = c['orig_paths']
    print(f'旧缓存记录 {len(old_paths)} 条路径\n')

    n_done = n_nofile = n_nomatch = n_same = 0
    miss = []
    for op in old_paths:
        new_path = idx.get(normalize_filename(os.path.basename(op)))
        if new_path is None:
            n_nomatch += 1
            if len(miss) < 10:
                miss.append(os.path.basename(op))
            continue
        old_npz = os.path.join(args.cache_dir, key_of(op) + '.npz')
        new_npz = os.path.join(args.cache_dir, key_of(new_path) + '.npz')
        if old_npz == new_npz:
            n_same += 1
            continue
        if not os.path.exists(old_npz):
            n_nofile += 1
            continue
        if not args.dry_run:
            if os.path.exists(new_npz):
                os.remove(new_npz)
            shutil.move(old_npz, new_npz)
        n_done += 1

    print(f'  已重命名     : {n_done}'
          f'{"   (dry-run，未实际移动)" if args.dry_run else ""}')
    print(f'  key 本来就对 : {n_same}')
    print(f'  旧 npz 缺失  : {n_nofile}')
    print(f'  库里找不到   : {n_nomatch}')
    if miss:
        print(f'    示例: {miss}')
    total = len([f for f in os.listdir(args.cache_dir) if f.endswith('.npz')])
    print(f'\n{args.cache_dir}/ 现有 npz: {total}')


if __name__ == '__main__':
    main()
