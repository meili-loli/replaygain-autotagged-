# -*- coding: utf-8 -*-
"""建立全库元数据（目录 / 专辑 / 艺人），给「组信号」特征用。

输出 library_meta.csv：file, path, dir_key, album, artist, source
  dir_key  规范化后的目录（用来分组；统一分隔符、去掉大小写歧义）
  album    优先取音频 tag，没有就用目录名
  artist   优先取音频 tag，没有就从文件名「歌曲-艺人-wy」里抽

读取 tag 是 IO 密集的，但只跑一次。已有结果会增量复用。
"""
import os
import re
import sys
import argparse
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'library_meta.csv')
AUDIO_EXTS = {'.flac', '.mp3', '.m4a', '.mp4', '.wav', '.ape', '.ogg', '.wv'}

try:
    import mutagen
    from mutagen.flac import FLAC
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, ID3NoHeaderError
    from mutagen.mp4 import MP4
except ImportError:
    sys.exit('请先 pip install mutagen')

WY_SUFFIX = re.compile(r'[-–—]\s*(wy|wyy|w)\s*$', re.I)


def norm_path(p):
    """跨格式统一的路径 key：斜杠统一、压缩重复、盘符小写其余保留大小写。"""
    q = str(p).replace('\\', '/')
    while '//' in q:
        q = q.replace('//', '/')
    if len(q) > 1 and q[1] == ':':
        q = q[0].lower() + q[1:]
    return q


def norm_dir(p):
    """规范化目录：统一分隔符、拼成稳定的 key。"""
    d = os.path.dirname(os.path.abspath(p))
    return norm_path(os.path.normpath(d))


def artist_from_name(name):
    stem = re.sub(r'\.(flac|mp3|m4a|wav|ape|ogg|wv)$', '', name, flags=re.I)
    stem = re.sub(r'^\d+[\s._-]+', '', stem)
    stem = WY_SUFFIX.sub('', stem).strip()
    parts = re.split(r'\s*[-–—]\s*', stem)
    if len(parts) >= 2:
        return parts[-1].strip()
    return ''


def read_tags(path):
    """返回 (album, artist)，读不到就是空串。"""
    ext = os.path.splitext(path)[1].lower()
    album = artist = ''
    try:
        if ext == '.flac':
            a = FLAC(path)
            album = str(a.get('album', [''])[0] if a.get('album') else '')
            artist = str(a.get('artist', [''])[0] if a.get('artist') else '')
        elif ext == '.mp3':
            try:
                t = ID3(path)
            except ID3NoHeaderError:
                t = None
            if t is not None:
                for k in ('TALB', 'TPE1'):
                    if k in t:
                        v = str(t[k].text[0]) if t[k].text else ''
                        if k == 'TALB':
                            album = v
                        else:
                            artist = v
        elif ext in ('.m4a', '.mp4', '.m4b', '.m4r'):
            a = MP4(path)
            if a.tags is not None:
                album = str(a.tags.get('\xa9alb', [''])[0])
                artist = str(a.tags.get('\xa9ART', [''])[0])
        else:
            a = mutagen.File(path, easy=True)
            if a is not None and a.tags is not None:
                album = str(a.tags.get('album', [''])[0])
                artist = str(a.tags.get('artist', [''])[0])
    except Exception:
        pass
    return album.strip(), artist.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='E:' + os.sep + '我的音乐')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--refresh', action='store_true')
    args = ap.parse_args()

    files = []
    for dp, _, fns in os.walk(args.root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                files.append(os.path.join(dp, fn))
    files.sort()
    if args.limit:
        files = files[:args.limit]
    print(f'扫描到 {len(files)} 个音频文件')

    old = {}
    if os.path.exists(OUT) and not args.refresh:
        try:
            od = pd.read_csv(OUT, encoding='utf-8-sig')
            old = {r['path']: (r['album'], r['artist'])
                   for _, r in od.iterrows()}
            print(f'复用已有元数据 {len(old)} 条')
        except Exception:
            pass

    rows = []
    n_tag = 0
    for i, p in enumerate(files):
        album, artist = old.get(p, ('', ''))
        src = 'cache'
        if not album and not artist:
            album, artist = read_tags(p)
            src = 'tag'
            n_tag += 1
        name = os.path.basename(p)
        if not artist:
            artist = artist_from_name(name)
            src = src + '+name' if src == 'tag' else 'name'
        if not album:
            album = os.path.basename(os.path.dirname(p))
            src = src + '+dir'
        rows.append(dict(file=name, path=p, path_key=norm_path(p),
                         dir_key=norm_dir(p), album=album, artist=artist,
                         source=src))
        if (i + 1) % 200 == 0:
            print(f'  [{i+1}/{len(files)}]')

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'\n写好 {OUT}  ({len(df)} 行，本次读 tag {n_tag} 个)')
    print()
    print('来源分布:')
    print(df.source.value_counts().to_string())
    print()
    print('分组规模（按 dir_key）:')
    vc = df.dir_key.value_counts()
    print(f'  目录数 {len(vc)}  最大组 {vc.iloc[0]} 首  单曲组 {int((vc==1).sum())} 个')
    print('分组规模（按 artist）:')
    va = df.artist.value_counts()
    print(f'  艺人数 {len(va)}  最大组 {va.iloc[0]} 首  单曲组 {int((va==1).sum())} 个')
    print()
    print('样例:')
    print(df.head(8)[['file', 'album', 'artist']].to_string(index=False))


if __name__ == '__main__':
    main()
