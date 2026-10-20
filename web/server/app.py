"""
replaygain-nyan 网页端后端
包装 replaygain-autotagged 真实脚本：曲库扫描 / 标注 CSV / 训练 / 写标签
运行：python server/app.py   （默认 http://127.0.0.1:8792）
配置：音乐目录写在 server_config.json 的 music_dir，或启动时 set RG_MUSIC_DIR
"""
import os
import sys
import csv
import json
import re
import time
import uuid
import hashlib
import io
import shutil
import zipfile
import tempfile
import urllib.request
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

WEB = Path(__file__).resolve().parent.parent          # .../web

# 真脚本目录（train_eval_export.py / predict_and_tag.py 所在处）。
# 历史布局是「web/ 与 replaygain-autotagged/ 平级」，所以原先写死成
# WEB.parent / "replaygain-autotagged"。但本仓库是「web/ 直接放在仓库根下」，
# 那样算出来会是 <repo>/replaygain-autotagged（不存在），导致
# 曲库扫描 / 训练 / 写标签全部失败。这里按「脚本是否存在」来探测，
# 两种布局都兼容。
_SCRIPT_DIR_CANDIDATES = [
    WEB.parent,                                  # 仓库根（当前布局）
    WEB.parent / "replaygain-autotagged",        # 旧布局：同级目录
    WEB.parent.parent,                           # web/ 在子目录里时的上一级
]
PROJECT = next(
    (p for p in _SCRIPT_DIR_CANDIDATES if (p / "predict_and_tag.py").is_file()),
    WEB.parent,
)

# 数据目录（delta_gain.csv / model_dir / feat_cache / library_meta.csv 所在处）。
# 脚本目录和数据目录**不一定在一起**：比如本机是「仓库在 ~/Music/replaygain-nyan，
# 而模型和标注都在 ~/Music」。所以单独探测：
#   1) 环境变量 RG_DATA 优先
#   2) 含 model_dir 或 delta_gain.csv 的候选
#   3) 兜底 = PROJECT（脚本目录就地存数据，干净 clone 的默认情形）
_DATA_DIR_CANDIDATES = [os.environ.get("RG_DATA") or ""] + [
    str(WEB.parent.parent),                      # 上一级（仓库外面的数据）
    str(WEB.parent),
    str(WEB.parent / "replaygain-autotagged"),
]
DATA = PROJECT
for _c in _DATA_DIR_CANDIDATES:
    if not _c:
        continue
    _p = Path(_c)
    if (_p / "model_dir").is_dir() or (_p / "delta_gain.csv").is_file():
        DATA = _p
        break

CSV_PATH = DATA / "delta_gain.csv"
CONFIG_PATH = WEB / "server_config.json"
LOG_DIR = WEB / "server_logs"
UPLOADS_ROOT = WEB / "uploads"

# 云端会话上限（可用环境变量覆盖）
SID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
MAX_FILE_MB = int(os.environ.get("RG_MAX_FILE_MB", "200"))
MAX_SESSION_MB = int(os.environ.get("RG_SESSION_MB", "1024"))
SESSION_DAYS = int(os.environ.get("RG_SESSION_DAYS", "7"))

# 鉴权：RG_ACCESS_CODE=应用访问码（两层防护的第二层，本地自用可不设）
#       CF_ACCESS_TEAM=Cloudflare Access 的 team 域（第一层，隧道部署时校验边缘传来的 JWT）
AUTH_CODE = os.environ.get("RG_ACCESS_CODE", "").strip()
AUTH_HASH = hashlib.sha256(AUTH_CODE.encode("utf-8")).hexdigest() if AUTH_CODE else ""
CF_ACCESS_TEAM = os.environ.get("CF_ACCESS_TEAM", "").strip()
CF_ACCESS_AUD = os.environ.get("CF_ACCESS_AUD", "").strip()
OPEN_PATHS = {"/api/auth", "/api/auth/status"}

# 模型市场：RG_HUB_URL 指向 models.json（GitHub raw / Releases 均可），默认连主仓库的 hub
_HUB_DEFAULT = "https://raw.githubusercontent.com/meili-loli/replaygain-autotagged-/main/hub/models.json"
HUB_URL = (os.environ.get("RG_HUB_URL") or _HUB_DEFAULT).strip()
HUB_DIR = WEB / "hub_models"

AUDIO_EXTS = {".flac", ".mp3", ".m4a", ".mp4", ".m4b", ".m4r",
              ".ogg", ".opus", ".wav", ".aiff", ".aif", ".ape", ".wv"}

app = FastAPI(title="replaygain-nyan")


@app.middleware("http")
async def no_cache(request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


def _code_ok(request: Request) -> bool:
    if not AUTH_CODE:
        return True
    if request.headers.get("x-access-code") == AUTH_CODE:
        return True
    return (request.cookies.get("rg_code") or "") == AUTH_HASH


def _cf_access_ok(request: Request) -> bool:
    """校验 Cloudflare Access 边缘传来的 JWT（CF_ACCESS_TEAM 形如 xxx.cloudflareaccess.com）"""
    tok = request.headers.get("cf-access-jwt-assertion") or ""
    if not tok:
        return False
    try:
        import jwt
        jwks = jwt.PyJWKClient(f"https://{CF_ACCESS_TEAM}/cdn-cgi/access/certs", cache_keys=True)
        key = jwks.get_signing_key_from_jwt(tok).key
        claims = jwt.decode(tok, key, algorithms=["RS256"],
                            issuer=f"https://{CF_ACCESS_TEAM}",
                            audience=CF_ACCESS_AUD or None,
                            options={"require": ["exp", "iss"]})
        return bool(claims)
    except Exception:
        return False


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    path = request.url.path
    if (path.startswith("/api/") or path.startswith("/music-files/")) and path not in OPEN_PATHS:
        if CF_ACCESS_TEAM and not _cf_access_ok(request):
            return JSONResponse({"detail": "Cloudflare Access 校验没过喵"}, status_code=401)
        if AUTH_CODE and not _code_ok(request):
            return JSONResponse({"detail": "需要访问码喵"}, status_code=401)
    return await call_next(request)


def load_cfg() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    env = os.environ.get("RG_MUSIC_DIR", "")
    return {"music_dir": env} if env else {}


def save_cfg(d: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def default_music_dir() -> str:
    d = WEB / "music"
    return str(d) if d.is_dir() else ""


def resolve_music_dir() -> str:
    """配置优先；没配置时回落到自带 music/ 示例曲库，换机器开箱即用"""
    d = (load_cfg().get("music_dir") or "").strip()
    if d:
        return d
    return default_music_dir()


def _sid(request: Request) -> str:
    v = request.cookies.get("rg_sid") or ""
    return v if SID_RE.fullmatch(v) else ""


def session_dir(request: Request):
    """云端会话目录：上传过歌才存在"""
    sid = _sid(request)
    if not sid:
        return None
    d = UPLOADS_ROOT / sid
    return d if d.is_dir() else None


def active_ctx(request: Request):
    """（曲库根目录, 标注 CSV, 模式）——云端会话优先，无会话回落本地配置"""
    sd = session_dir(request)
    if sd is not None:
        return sd, sd / "delta_gain.csv", "cloud"
    return Path(resolve_music_dir()), CSV_PATH, "local"


def music_root(request: Request) -> str:
    root, _, _ = active_ctx(request)
    if not root or not os.path.isdir(str(root)):
        raise HTTPException(400, "还没设置音乐目录喵：先在曲库点「更换音乐目录」填写路径")
    return str(root)


# ============================================================
# 鉴权：访问码登录 / 状态查询
# ============================================================
@app.get("/api/auth/status")
def api_auth_status(request: Request):
    return {"required": bool(AUTH_CODE), "authed": _code_ok(request)}


@app.post("/api/auth")
def api_auth(body: dict):
    if not AUTH_CODE:
        return {"ok": True}
    if str(body.get("code") or "") != AUTH_CODE:
        raise HTTPException(401, "访问码不对喵")
    resp = JSONResponse({"ok": True})
    resp.set_cookie("rg_code", AUTH_HASH, max_age=30 * 86400,
                    httponly=True, samesite="lax")
    return resp


# ============================================================
# 配置
# ============================================================
@app.get("/api/config")
def api_get_config(request: Request):
    d = resolve_music_dir()
    _, _, mode = active_ctx(request)
    return {"music_dir": d, "valid": bool(d) and os.path.isdir(d), "mode": mode}


@app.post("/api/config")
def api_set_config(body: dict):
    d = (body.get("music_dir") or "").strip()
    if not os.path.isdir(d):
        raise HTTPException(400, f"目录不存在喵：{d}")
    save_cfg({"music_dir": d})
    return {"ok": True, "music_dir": d}


# ============================================================
# 标注 CSV（file,delta_gain —— 与 annotate_gain.py 完全同格式）
# ============================================================
def read_marks(csv_path: Path = CSV_PATH) -> dict:
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            fn = (r.get("file") or "").strip()
            g = (r.get("delta_gain") or "").strip()
            if fn and g:
                try:
                    out[fn] = float(g)
                except ValueError:
                    pass
    return out


def write_marks(marks: dict, csv_path: Path = CSV_PATH) -> None:
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "delta_gain"])
        for fn, g in marks.items():
            w.writerow([fn, f"{g:.2f}"])


@app.get("/api/marks")
def api_marks(request: Request):
    return read_marks(active_ctx(request)[1])


@app.post("/api/mark")
def api_mark(request: Request, body: dict):
    file = (body.get("file") or "").strip()
    gain = body.get("gain")
    if not file or gain is None:
        raise HTTPException(400, "缺 file 或 gain 喵")
    csv_path = active_ctx(request)[1]
    marks = read_marks(csv_path)
    marks[file] = round(float(gain), 2)
    write_marks(marks, csv_path)
    return {"ok": True, "file": file, "gain": marks[file], "total": len(marks)}


@app.post("/api/mark/delete")
def api_mark_delete(request: Request, body: dict):
    file = (body.get("file") or "").strip()
    csv_path = active_ctx(request)[1]
    marks = read_marks(csv_path)
    if file in marks:
        del marks[file]
        write_marks(marks, csv_path)
    return {"ok": True}


# ============================================================
# 曲库扫描
# ============================================================
@app.get("/api/library")
def api_library(request: Request, refresh: int = 0):
    root, csv_path, mode = active_ctx(request)
    if not root or not os.path.isdir(str(root)):
        raise HTTPException(400, "还没设置音乐目录喵")
    marks = read_marks(csv_path)
    out = []
    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames.sort()
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in AUDIO_EXTS:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, str(root)).replace("\\", "/")
            fmt = ext[1:].upper()
            duration = None
            try:
                import mutagen
                m = mutagen.File(full)
                if m is not None and m.info is not None:
                    duration = round(getattr(m.info, "length", 0) or 0, 1)
            except Exception:
                pass
            g = marks.get(fn)
            out.append({
                "no": f"{len(out) + 1:03d}",
                "file": fn,
                "rel": rel,
                "fmt": fmt,
                "duration": duration,
                "mark": g,
                "state": "READY" if g is not None else "待标注",
            })
    out.sort(key=lambda t: t["file"])
    for i, t in enumerate(out):
        t["no"] = f"{i + 1:03d}"
    return {"total": len(out), "marked": sum(1 for t in out if t["mark"] is not None),
            "mode": mode, "tracks": out}


JOBS = {}


def _jkey(request: Request, name: str):
    return (_sid(request) or "local", name)


def _start_job(key, cmd: list) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{key[1]}-{key[0]}.log"
    lf = open(log_path, "wb")
    try:
        lf.write(f"$ {' '.join(cmd)}\n\n".encode("utf-8"))
        lf.flush()
        # Popen 会复制句柄，父进程随后关闭安全
        proc = subprocess.Popen(cmd, cwd=str(DATA), stdout=lf,
                                stderr=subprocess.STDOUT)
    finally:
        lf.close()
    JOBS[key] = {"proc": proc, "log": str(log_path), "start": log_path}


def _job_state(key, tail_lines: int = 6) -> dict:
    job = JOBS.get(key)
    if job is None:
        return {"running": False, "started": False, "tail": []}
    proc = job["proc"]
    running = proc.poll() is None
    tail = []
    try:
        lines = Path(job["log"]).read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-tail_lines:]
    except Exception:
        pass
    mae = None
    try:
        txt = Path(job["log"]).read_text(encoding="utf-8", errors="replace")
        import re
        m = re.findall(r"MAE[^0-9]{0,6}([0-9.]+)", txt)
        if m:
            mae = m[-1]
    except Exception:
        pass
    return {"running": running, "started": True, "exit": proc.poll(),
            "tail": tail, "mae": mae}


@app.post("/api/train/start")
def api_train_start(request: Request, body: dict = {}):
    root, csv_path, mode = active_ctx(request)
    if not root or not os.path.isdir(str(root)):
        raise HTTPException(400, "先设置音乐目录喵")
    if not read_marks(csv_path):
        raise HTTPException(400, "还没有任何 mark —— 先去 ① 标注台标几首喵")
    key = _jkey(request, "train")
    job = JOBS.get(key)
    if job and job["proc"].poll() is None:
        return {"ok": True, "already": True}
    py = sys.executable
    model_dir = (root / "model_dir") if mode == "cloud" else (DATA / "model_dir")
    cmd = [py, str(PROJECT / "train_eval_export.py"),
           "--csv", str(csv_path), "--audio-root", str(root),
           "--model-dir", str(model_dir), "--n-jobs", "4"]
    if mode == "cloud":
        cmd += ["--cache", str(root / "feat_cache.bin"),
                "--eval-dir", str(root / "eval")]
    workers = os.environ.get("RG_WORKERS", "").strip()
    if workers:
        cmd += ["--n-jobs", workers]  # 小内存主机限制训练并发，防 OOM
    if body.get("group_signal", True):
        cmd.append("--group-signal")
    _start_job(key, cmd)
    return {"ok": True}


@app.get("/api/train/status")
def api_train_status(request: Request):
    return _job_state(_jkey(request, "train"))


@app.post("/api/write/start")
def api_write_start(request: Request, body: dict = {}):
    root, _, mode = active_ctx(request)
    if not root or not os.path.isdir(str(root)):
        raise HTTPException(400, "先设置音乐目录喵")
    model_dir = (root / "model_dir") if mode == "cloud" else (DATA / "model_dir")
    md = str(body.get("model_dir") or "").strip()
    if md:
        p = Path(md).resolve()
        allowed = [HUB_DIR.resolve()]
        if mode == "cloud":
            allowed.append(Path(root).resolve())
        if not any(str(p) == str(a) or str(p).startswith(str(a) + os.sep)
                   for a in allowed):
            raise HTTPException(400, "只能选用 hub 下载的模型或本会话训练的模型喵")
        if not (p / "meta.json").is_file():
            raise HTTPException(400, "那个模型目录里没有 meta.json 喵")
        model_dir = p
    py = sys.executable
    cmd = [py, str(PROJECT / "predict_and_tag.py"),
           "--input", str(root), "--model-dir", str(model_dir),
           "--log-dir", str((root / "logs") if mode == "cloud" else LOG_DIR)]
    if mode == "cloud":
        cmd += ["--feat-cache", str(root / "feat_cache")]
    workers = os.environ.get("RG_WORKERS", "").strip()
    if workers:
        cmd += ["--workers", workers]  # 小内存主机限制推理并发，防 OOM
    if body.get("dry_run", True):
        cmd.append("--dry-run")
    if body.get("skip_tagged", True):
        cmd.append("--skip-tagged")
    key = _jkey(request, "write")
    _start_job(key, cmd)
    return {"ok": True}


@app.get("/api/write/status")
def api_write_status(request: Request):
    return _job_state(_jkey(request, "write"))


# ============================================================
# 前端静态 + 音频文件（支持 Range，浏览器可拖进度）
# ============================================================
@app.get("/music-files/{relpath:path}")
def api_music_file(request: Request, relpath: str):
    root = Path(music_root(request)).resolve()
    full = (root / relpath).resolve()
    if not str(full).startswith(str(root)) or not full.is_file():
        raise HTTPException(404, "文件不存在喵")
    media = {".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg",
             ".opus": "audio/ogg", ".m4a": "audio/mp4", ".wav": "audio/wav"}
    ext = full.suffix.lower()
    return FileResponse(str(full), media_type=media.get(ext, "application/octet-stream"))


# ============================================================
# 云端会话：上传 / 打包下载 / 退出
# ============================================================
@app.post("/api/session/upload")
async def api_session_upload(request: Request, files: list[UploadFile] = File(...)):
    sid = _sid(request) or str(uuid.uuid4())
    d = UPLOADS_ROOT / sid
    d.mkdir(parents=True, exist_ok=True)

    used_total = sum(p.stat().st_size for p in d.rglob("*")
                     if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    fcap = MAX_FILE_MB * 1024 * 1024
    scap = MAX_SESSION_MB * 1024 * 1024
    saved, skipped = [], []
    for f in files:
        fn = os.path.basename((f.filename or "").replace("\\", "/"))
        ext = os.path.splitext(fn)[1].lower()
        if not fn or ext not in AUDIO_EXTS:
            skipped.append(f.filename or "未知文件")
            continue
        size = 0
        over = False
        dest = d / fn
        with open(dest, "wb") as out:
            while True:
                chunk = await f.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > fcap or used_total + size > scap:
                    over = True
                    break
                out.write(chunk)
        if over:
            dest.unlink(missing_ok=True)
            skipped.append(f"{fn}（超{'单文件' if size > fcap else '会话'}上限）")
        else:
            used_total += size
            saved.append(fn)
    resp = JSONResponse({"ok": True, "saved": saved, "skipped": skipped, "sid": sid})
    resp.set_cookie("rg_sid", sid, max_age=SESSION_DAYS * 86400,
                    httponly=True, samesite="lax")
    return resp


@app.get("/api/session/download")
def api_session_download(request: Request):
    sd = session_dir(request)
    if sd is None:
        raise HTTPException(400, "还没有云端会话喵：先上传几首歌")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sd.rglob("*")):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                z.write(p, p.name)
    tmp.close()
    return FileResponse(tmp.name, media_type="application/zip",
                        filename="replaygain-tagged.zip",
                        background=BackgroundTask(os.unlink, tmp.name))


@app.post("/api/session/reset")
def api_session_reset(request: Request):
    sid = _sid(request)
    sd = session_dir(request)
    if sd is not None:
        shutil.rmtree(sd, ignore_errors=True)
    for key in [k for k in JOBS if k[0] == sid]:
        proc = JOBS.pop(key)["proc"]
        if proc.poll() is None:
            proc.terminate()
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("rg_sid")
    return resp


@app.on_event("startup")
def _prune_expired_sessions():
    if not UPLOADS_ROOT.is_dir():
        return
    cutoff = time.time() - SESSION_DAYS * 86400
    for d in UPLOADS_ROOT.iterdir():
        if d.is_dir() and SID_RE.fullmatch(d.name) and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


# ============================================================
# 模型市场：模型卡 / 打包 / hub 浏览与下载
# ============================================================
def _local_feature_pipeline_version() -> int:
    """从训练脚本读 CACHE_VERSION，作为本地特征管线版本（hub 兼容性比对用）。"""
    try:
        src = (PROJECT / "train_eval_export.py").read_text(encoding="utf-8",
                                                           errors="replace")
        m = re.search(r"CACHE_VERSION\s*=\s*(\d+)", src)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


LOCAL_FPV = _local_feature_pipeline_version()
_HUB_CACHE = {"at": 0.0, "items": None}


def _model_dir_for(request: Request):
    root, _, mode = active_ctx(request)
    return (root / "model_dir") if mode == "cloud" else (DATA / "model_dir")


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract_model_zip(zf: zipfile.ZipFile, dest: Path):
    """只允许 .txt / meta.json / MODEL_CARD.json，拒绝绝对路径和 ..。"""
    for m in zf.namelist():
        name = m.replace("\\", "/")
        if name.startswith("/") or ".." in name.split("/"):
            raise HTTPException(400, f"压缩包里有可疑路径喵：{m}")
        base = os.path.basename(name)
        if base != "meta.json" and base != "MODEL_CARD.json" and not base.endswith(".txt"):
            raise HTTPException(400, f"模型包里有不该有的文件喵：{m}（只允许 .txt / meta.json / MODEL_CARD.json）")
    zf.extractall(dest)


def _validate_model_dir(d: Path):
    """加载前的最后一道门：meta 必须在、必须是原生格式、特征版本必须匹配。"""
    meta_p = d / "meta.json"
    if not meta_p.is_file():
        raise HTTPException(400, "模型包里没有 meta.json 喵")
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(400, "meta.json 不是合法 JSON 喵")
    has_txt = any(p.suffix == ".txt" for p in d.rglob("*.txt") if p.is_file())
    if not has_txt:
        raise HTTPException(400, "模型包里没有原生 .txt 模型文件喵（pickle 模型不能用于共享）")
    fpv = meta.get("feature_pipeline_version")
    if fpv is not None and int(fpv) != LOCAL_FPV:
        raise HTTPException(400, f"模型特征管线版本 v{int(fpv)} 与本地 v{LOCAL_FPV} 不匹配喵")
    return meta


@app.get("/api/model/current")
def api_model_current(request: Request):
    model_dir = _model_dir_for(request)
    meta_p = model_dir / "meta.json"
    card_p = model_dir / "MODEL_CARD.json"
    if not meta_p.exists():
        return {"exists": False, "mode": active_ctx(request)[2]}
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    card = json.loads(card_p.read_text(encoding="utf-8")) if card_p.exists() else None
    return {"exists": True, "mode": active_ctx(request)[2],
            "model_dir": str(model_dir), "meta": meta, "card": card}


@app.post("/api/model/pack")
def api_model_pack(request: Request, body: dict = {}):
    root, csv_path, mode = active_ctx(request)
    model_dir = _model_dir_for(request)
    if not (model_dir / "meta.json").exists():
        raise HTTPException(400, "还没有能发布的模型喵：先去 ③ 训练")
    # 打包逻辑全部在 model_export.py（autotagged 侧）：转换原生格式 + 模型卡 + zip
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    cmd = [sys.executable, str(PROJECT / "model_export.py"), "pack",
           "--model-dir", str(model_dir),
           "--csv", str(csv_path),
           "--out", tmp.name,
           "--author", str(body.get("author") or "")[:64],
           "--note", str(body.get("note") or "")[:200]]
    if root and os.path.isdir(str(root)):
        cmd += ["--audio-root", str(root)]
    if body.get("publish_tracks"):
        cmd.append("--publish-tracks")
    try:
        r = subprocess.run(cmd, cwd=str(DATA), capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        os.unlink(tmp.name)
        raise HTTPException(400, "打包超时喵：曲库太大，歌单公开先关掉试试")
    if r.returncode != 0 or not os.path.isfile(tmp.name) or os.path.getsize(tmp.name) == 0:
        err = (r.stderr or r.stdout or b"").decode("utf-8", "replace").strip().splitlines()
        os.unlink(tmp.name)
        raise HTTPException(400, err[-1] if err else "打包失败喵")
    name = f"rg-model-{str(body.get('author') or 'anonymous')[:32]}-{time.strftime('%Y%m%d')}.zip"
    return FileResponse(tmp.name, media_type="application/zip", filename=name,
                        headers={"X-Model-Sha256": _sha256_file(tmp.name),
                                 "X-Model-Name": name},
                        background=BackgroundTask(os.unlink, tmp.name))


def _load_hub_index(force: bool = False):
    now = time.time()
    if force or not _HUB_CACHE["items"] or now - _HUB_CACHE["at"] > 3600:
        with urllib.request.urlopen(HUB_URL, timeout=10) as r:
            idx = json.loads(r.read().decode("utf-8"))
        items = idx if isinstance(idx, list) else idx.get("models", [])
        _HUB_CACHE.update(at=now, items=items)
    return _HUB_CACHE["items"] or []


@app.get("/api/hub/models")
def api_hub_models(force: int = 0):
    if not HUB_URL:
        return {"configured": False, "local_fpv": LOCAL_FPV, "models": []}
    if force or not _HUB_CACHE["items"] or time.time() - _HUB_CACHE["at"] > 3600:
        try:
            _load_hub_index(force=force)
        except Exception as e:
            if not _HUB_CACHE["items"]:
                return {"configured": True, "local_fpv": LOCAL_FPV,
                        "error": f"hub 拉取失败喵：{e}", "models": []}
    out = []
    for it in (_HUB_CACHE["items"] or []):
        it = dict(it)
        try:
            it["compatible"] = (it.get("feature_pipeline_version") is None
                                or int(it["feature_pipeline_version"]) == LOCAL_FPV)
        except Exception:
            it["compatible"] = False
        it["downloaded"] = (HUB_DIR / str(it.get("id", ""))).is_dir()
        out.append(it)
    return {"configured": True, "local_fpv": LOCAL_FPV, "models": out}


@app.post("/api/hub/download")
def api_hub_download(request: Request, body: dict):
    if not HUB_URL:
        raise HTTPException(400, "hub 没有配置喵（设 RG_HUB_URL 环境变量）")
    item_id = str(body.get("id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", item_id):
        raise HTTPException(400, "模型 id 不合法喵")
    item = next((i for i in (_HUB_CACHE["items"] or [])
                 if str(i.get("id")) == item_id), None)
    if item is None:
        # 缓存里没有（比如刚重启）就现场拉一次索引再找
        try:
            item = next((i for i in _load_hub_index(force=True)
                         if str(i.get("id")) == item_id), None)
        except Exception as e:
            raise HTTPException(400, f"hub 拉取失败喵：{e}")
    if item is None:
        raise HTTPException(404, "hub 索引里没有这个模型喵")
    url = item.get("download_url") or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()  # Windows：句柄不关掉 unlink 会失败
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp.name, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        os.unlink(tmp.name)
        raise HTTPException(400, f"下载失败喵：{e}")
    want = str(item.get("sha256") or "").lower()
    got = _sha256_file(tmp.name)
    if want and got != want:
        os.unlink(tmp.name)
        raise HTTPException(400, "sha256 校验不一致喵：这个包被篡改或索引过期了，刷新后重试")
    dest = HUB_DIR / item_id
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(tmp.name) as zf:
            _safe_extract_model_zip(zf, dest)
        _validate_model_dir(dest)
    except HTTPException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    finally:
        os.unlink(tmp.name)
    return {"ok": True, "id": item_id, "model_dir": str(dest)}


@app.post("/api/hub/import")
async def api_hub_import(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "不是合法的 zip 喵")
    item_id = uuid.uuid4().hex[:12]
    dest = HUB_DIR / item_id
    dest.mkdir(parents=True, exist_ok=True)
    try:
        _safe_extract_model_zip(zf, dest)
        meta = _validate_model_dir(dest)
    except HTTPException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    card_p = dest / "MODEL_CARD.json"
    card = json.loads(card_p.read_text(encoding="utf-8")) if card_p.exists() else {}
    return {"ok": True, "id": item_id,
            "name": card.get("model_name") or item_id,
            "model_type": meta.get("model_type")}


@app.get("/api/model/list-local")
def api_model_list_local():
    out = []
    if HUB_DIR.is_dir():
        for d in sorted(HUB_DIR.iterdir()):
            mp_ = d / "meta.json"
            if not d.is_dir() or not mp_.is_file():
                continue
            try:
                meta = json.loads(mp_.read_text(encoding="utf-8"))
            except Exception:
                continue
            cp = d / "MODEL_CARD.json"
            card = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
            out.append({"id": d.name, "name": card.get("model_name") or d.name,
                        "author": card.get("author", ""),
                        "model_type": meta.get("model_type"),
                        "samples": card.get("sample_count"),
                        "mae": (card.get("oof_stats") or {}).get("mae"),
                        "model_dir": str(d),
                        "compatible": (meta.get("feature_pipeline_version") is None
                                       or int(meta["feature_pipeline_version"]) == LOCAL_FPV)})
    return {"models": out}


app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("RG_HOST", "127.0.0.1")
    # Render 等平台用 PORT 环境变量指定监听端口
    port = int(os.environ.get("RG_PORT") or os.environ.get("PORT") or "8792")
    print(f"replaygain-nyan 网页端 → http://127.0.0.1:{port}  (bind {host})")
    uvicorn.run(app, host=host, port=port, log_level="warning")
