import os
import sqlite3
import threading
import time
import json
import hashlib
import shutil
import subprocess
from datetime import datetime
from typing import Optional, List, Tuple
import sys
from pathlib import Path
import re
_D = Path(__file__).resolve().parent
_R = _D.parent
pp = str(_R)
if pp not in sys.path:
    sys.path.insert(0, pp)

DB_PATH = os.path.join(os.path.abspath("."), "data", "tasks.db")
DATA_DIR = os.path.join(os.path.abspath("."), "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v is not None and len(v) > 0 else default

def _mask(s: Optional[str]) -> str:
    if not s:
        return ""
    n = len(s)
    if n <= 6:
        return "*" * n
    return s[:3] + "*" * (n - 6) + s[-3:]

def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def _connect():
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def _init_db():
    conn = _connect()
    cur = conn.cursor()
    if not _table_exists(conn, "tasks"):
        cur.execute(
            """
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                share_link TEXT,
                extract_code TEXT,
                status TEXT,
                progress REAL,
                created_at DATETIME,
                started_at DATETIME,
                finished_at DATETIME,
                result_video_path TEXT,
                cover_image_path TEXT,
                error_message TEXT
            )
            """
        )
    if not _table_exists(conn, "task_logs"):
        cur.execute(
            """
            CREATE TABLE task_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                timestamp DATETIME,
                level TEXT,
                message TEXT
            )
            """
        )
    if not _table_exists(conn, "files"):
        cur.execute(
            """
            CREATE TABLE files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                path TEXT,
                type TEXT,
                size_bytes INTEGER,
                ordered_index INTEGER
            )
            """
        )
    if not _table_exists(conn, "setting"):
        cur.execute(
            """
            CREATE TABLE setting (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME
            )
            """
        )
        init_settings_db()
        
    conn.commit()
    conn.close()

def _del_data():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM tasks")
    cur.execute("DELETE FROM task_logs")
    cur.execute("DELETE FROM files")
    conn.commit()
    conn.close()
_init_db()

def _setup_ffmpeg_env():
    if os.name == "nt":
        p = os.path.join(os.path.abspath("."), "ffmpeg", "bin", "ffmpeg.exe")
    else:
        p = os.path.join(os.path.abspath("."), "ffmpeg", "bin", "ffmpeg")
    
    if os.path.exists(p):
        os.environ["IMAGEIO_FFMPEG_EXE"] = p
        os.environ["FFMPEG_BINARY"] = p
        print(f"已设置 FFMPEG 环境变量: {p}")
    else:
        print(f"警告: 未找到内置 FFmpeg: {p}")

_setup_ffmpeg_env()

# _del_data()
from videoautoclip.settings import init_settings_db, load_settings_into_env, get_all_settings, save_settings
from videoautoclip.utils.asrutil import qwen_asr_flash_async_recog
load_settings_into_env()

def _log(task_id: int, level: str, message: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO task_logs(task_id,timestamp,level,message) VALUES(?,?,?,?)",
        (task_id, datetime.now(), level, message),
    )
    conn.commit()
    conn.close()

def _update_task(task_id: int, **kv):
    if not kv:
        return
    keys = ",".join([f"{k}=?" for k in kv.keys()])
    vals = list(kv.values())
    conn = _connect()
    cur = conn.cursor()
    cur.execute(f"UPDATE tasks SET {keys} WHERE id=?", (*vals, task_id))
    conn.commit()
    conn.close()

def _insert_file(task_id: int, path: str, ftype: str, size_bytes: int, idx: int):
    conn = _connect()
    cur = conn.cursor()
    #先查询是否存在
    cur.execute(
        "SELECT id FROM files WHERE task_id=? AND path=?",
        (task_id, path),
    )
    if cur.fetchone() is not None:
        conn.close()
        return
    cur.execute(
        "INSERT INTO files(task_id,path,type,size_bytes,ordered_index) VALUES(?,?,?,?,?)",
        (task_id, path, ftype, size_bytes, idx),
    )
    conn.commit()
    conn.close()

def create_task(payload: dict) -> int:
    title = payload.get("title") or ""
    share_link = payload.get("share_link") or ""
    extract_code = payload.get("extract_code") or ""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tasks(title,share_link,extract_code,status,progress,created_at) VALUES(?,?,?,?,?,?)",
        (title, share_link, extract_code, "queued", 0.0, datetime.now()),
    )
    task_id = cur.lastrowid
    conn.commit()
    conn.close()
    _log(task_id, "INFO", f"创建任务 {task_id} {title}")
    return task_id

def get_tasks(limit: int = 100) -> List[dict]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id,title,share_link,extract_code,status,progress,created_at,finished_at FROM tasks ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_task_logs(task_id: int, limit: int = 500) -> List[dict]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp,level,message FROM task_logs WHERE task_id=? ORDER BY id ASC LIMIT ?",
        (task_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def cancel_task(task_id: int):
    _update_task(task_id, status="cancelled", finished_at=datetime.now())
    _log(task_id, "INFO", "任务取消")

def update_progress(task_id: int, p: float):
    _update_task(task_id, progress=float(max(0.0, min(100.0, p))))

def _task_dir(task_id: int) -> str:
    p = os.path.join(DATA_DIR, "tasks", str(task_id))
    os.makedirs(p, exist_ok=True)
    return p

def _safe_filename(name: str) -> str:
    base = os.path.basename(name)
    base = base.split("?")[0]
    base = "".join([c if c.isalnum() or c in "._-" else "_" for c in base])
    return base or f"file_{hashlib.md5(name.encode('utf-8')).hexdigest()}"

def _resolve_local_path(work_dir: str, name: str) -> str:
    name = os.path.basename(name)
    p_sanitized = os.path.join(work_dir, _safe_filename(name))
    p_original = os.path.join(work_dir, name)
    if os.path.exists(p_sanitized):
        return p_sanitized
    if os.path.exists(p_original):
        return p_original
    return p_sanitized

def _cmd():
    return "BaiduPCS-Py"

def _ffmpeg_bin() -> str:
    #如果是windows 需要使用ffmpeg.exe
    if os.name == "nt":
        p = os.path.join(os.path.abspath("."), "ffmpeg", "bin", "ffmpeg.exe")
    else:
        p = os.path.join(os.path.abspath("."), "ffmpeg", "bin", "ffmpeg")
    print(f"FFMPEG 路径: {p}")
    if os.path.exists(p):
        return p
    return os.environ.get("FFMPEG_BIN", "ffmpeg")

def _decode_output(b: bytes) -> str:
    for enc in ("utf-8", "gbk", "cp936", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            pass
    return b.decode("utf-8", errors="replace")

# def _run(args: List[str], capture: bool = True, timeout: Optional[int] = None) -> str:
#     exe = _cmd()
#     env = os.environ.copy()
#     env["PYTHONIOENCODING"] = "utf-8"
#     to = timeout if timeout is not None else (None if (args and args[0] == "download") else 60)
#     if capture:
#         print(f"运行命令: {exe} {' '.join(args)}")
#         p = subprocess.run([exe] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=to, env=env, text=False)
#         return _decode_output(p.stdout or b"")
#     else:
#         subprocess.run([exe] + args, check=True, stdin=subprocess.DEVNULL, timeout=to, env=env)
#         return ""

def _run(args, capture=True, input_data=None):
    exe = _cmd()
    print(f"运行: {exe} {' '.join(args)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    to = None if (args and args[0] == "download") else 60
    
    run_kwargs = {
        "timeout": to,
        "env": env,
    }
    if input_data:
        run_kwargs["input"] = input_data.encode("utf-8")
    else:
        run_kwargs["stdin"] = subprocess.DEVNULL

    if capture:
        run_kwargs["stdout"] = subprocess.PIPE
        run_kwargs["stderr"] = subprocess.STDOUT
        run_kwargs["text"] = False
        p = subprocess.run([exe] + args, **run_kwargs)
        b = p.stdout or b""
        def _decode_output(data):
            for enc in ("utf-8", "gbk", "cp936", "latin-1"):
                try:
                    return data.decode(enc)
                except Exception:
                    pass
            return data.decode("utf-8", errors="replace")
        s = _decode_output(b)
        print(f"输出: {s}")
        return s
    else:
        run_kwargs["check"] = True
        subprocess.run([exe] + args, **run_kwargs)
        return ""

def _parse_ls_dirs(text: str) -> List[str]:
    res: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue

        m = re.match(r"^[dD](?:\||\s+)(.*)$", s)
        if m:
            res.append(m.group(1).strip())
    if res:
        return res
    res2: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith(("name", "total")):
            continue
        if s.startswith("/") or s.startswith("|") or s.startswith("-"):
            continue
        if "|" in s:
            s = s.split("|", 1)[-1].strip()
        parts = s.split()
        name = parts[-1] if parts else s
        if name != "Path":
            res2.append(name)
    return res2

def _parse_ls_names(text: str) -> List[str]:
    res: List[str] = []
    
    lines = text.splitlines()
    root_path = None
    current_path = None

    for line in lines:
        s = line.strip()
        if not s:
            continue
        
        # 识别目录头 /VideoAutoClip/...
        if s.startswith("/"):
            if root_path is None:
                root_path = s
            current_path = s
            continue

        if s.lower().startswith(("name", "total", "path")):
            continue
        if re.match(r"^[\-\s\u2500]+$", s):
            continue
            
        nm = None
        # 匹配 d/f/- 开头的行
        m = re.match(r"^[dDfF\-](?:\||\s+)(.*)$", s)
        if m:
            nm = m.group(1).strip()
        else:
            # 兼容其他格式
            if s.startswith("|"):
                continue
            if "|" in s:
                nm = s.split("|", 1)[-1].strip()
            else:
                nm = s
        
        if nm and nm != "Path":
            prefix = ""
            if root_path and current_path and current_path.startswith(root_path):
                rel = current_path[len(root_path):]
                if rel.startswith("/"):
                    rel = rel[1:]
                if rel:
                    prefix = rel + "/"
            res.append(prefix + nm)
    return res

def _is_video_valid(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return False
    # 尝试读取前0.1秒来验证是否可解码
    cmd = [_ffmpeg_bin(), "-v", "error", "-i", path, "-t", "0.1", "-f", "null", "-"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def _concatenate_videos(video_paths: List[str], output_path: str):
    if not video_paths:
        print("无视频文件可合并")
        return
    #去除video_paths重复项
    video_paths = list(set(video_paths))
    print(f"去除重复{video_paths}")
    #重新文件名的按升序排序
    video_paths.sort(key=lambda x: [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', x)])

    print(f"排序后{video_paths}")
    output_path = os.path.abspath(output_path)
    rnd = f"video_list_{hashlib.md5(('|'.join(video_paths)).encode('utf-8')).hexdigest()}.txt"
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    tmp_list = os.path.join(out_dir, rnd)
    print(f"生成合并列表文件: {tmp_list}")
    try:
        # 使用 'w' 模式覆盖写入，确保不会追加到旧文件
        with open(tmp_list, "w", encoding="utf-8") as f:
            for p in video_paths:
                # ffmpeg concat protocol requires escaping or forward slashes on Windows
                safe_p = os.path.abspath(p).replace("\\", "/")
                f.write(f"file '{safe_p}'\n")
    except Exception as e:
        print(f"写入列表文件失败: {e}")
        raise

    cmd = [_ffmpeg_bin(), "-f", "concat", "-safe", "0", "-i", tmp_list, "-c", "copy", "-movflags", "+faststart", "-y", output_path]
    print(f"执行合并命令: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        if not _is_video_valid(output_path):
             raise Exception("合并后文件验证失败")
    except Exception as e:
        print(f"快速合并失败，尝试重编码合并: {e}")
        # 移除 -c copy 进行重编码合并
        cmd = [_ffmpeg_bin(), "-f", "concat", "-safe", "0", "-i", tmp_list, "-movflags", "+faststart", "-y", output_path]
        try:
             subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e2:
             print(f"重编码合并也失败: {e2}")
             raise e2
    finally:
        if os.path.exists(tmp_list):
            # os.remove(tmp_list) # 暂时保留以便调试
            pass

def _export_audio(video_path: str, audio_output_path: str):
    os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
    cmd = [_ffmpeg_bin(), "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_output_path]
    subprocess.run(cmd, check=True)

def _pick_cover_and_videos(files: List[str]) -> Tuple[Optional[str], List[str]]:
    imgs = sorted([p for p in files if os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png", ".webp"]])
    vids = sorted([p for p in files if os.path.splitext(p)[1].lower() in [".mp4", ".mov", ".mkv", ".avi", ".m4v"]])
    cover = imgs[0] if imgs else None
    return cover, vids[:3]

def _ensure_pcs_user():
    cookies = _env("BAIDU_PCS_COOKIES", "")
    bduss = _env("BAIDU_PCS_BDUSS", "")
    exe = _cmd()
    if not exe:
        return "未找到BaiduPCS-Py"
    if cookies and bduss:
        try:
            rnd_user = f"u{int(time.time())}\n"
            p = subprocess.run([exe, "useradd", "--cookies", cookies, "--bduss", bduss], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, input=rnd_user.encode('utf-8'), timeout=20, env=os.environ.copy())
            
            def _decode(b):
                for enc in ("utf-8", "gbk", "cp936", "latin-1"):
                    try:
                        return b.decode(enc)
                    except:
                        pass
                return b.decode("utf-8", errors="replace")
            
            out_str = _decode(p.stdout or b"")
            return "已设置BaiduPCS用户" if p.returncode == 0 else f"设置失败: {out_str.strip()}"
        except Exception as e:
            return f"设置失败: {e}"
    return "缺少Cookie/BDUSS"

_active_lock = threading.Lock()
_active_running: set = set()

def run_scheduler_tick():
    max_c = int(_env("MAX_CONCURRENT_TASKS", "1") or "1")
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id FROM tasks WHERE status='running'")
    running_ids = [r[0] for r in cur.fetchall()]
    conn.close()
    with _active_lock:
        current = len(running_ids) + len(_active_running)
        if current >= max_c:
            return
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id,share_link FROM tasks WHERE status='queued' ORDER BY created_at ASC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if not row:
        return
    task_id = row[0]
    with _active_lock:
        _active_running.add(task_id)
    t = threading.Thread(target=_execute_task, args=(task_id,), daemon=True)
    t.start()

def _execute_task(task_id: int):
    try:
        _update_task(task_id, status="running", started_at=datetime.now(), progress=0.0)
        cookies = _env("BAIDU_PCS_COOKIES", "")
        bduss = _env("BAIDU_PCS_BDUSS", "")
        _log(task_id, "INFO", f"DASHSCOPE_API_KEY={_mask(_env('DASHSCOPE_API_KEY',''))} MODEL={_env('MODEL_NAME','qwen-plus')} COOKIES={_mask(cookies)} BDUSS={_mask(bduss)}")
        msg = _ensure_pcs_user()
        _log(task_id, "INFO", msg)
        update_progress(task_id, 2)
        conn = _connect()
        cur = conn.cursor()
        cur.execute("SELECT share_link FROM tasks WHERE id=?", (task_id,))
        r = cur.fetchone()
        conn.close()
        link = r[0] if r else ""
        #从link中提取pwd
        pwd = ""
        m = re.search(r"pwd=([^&]+)", link)
        if m:
            pwd = m.group(1)
            _log(task_id, "INFO", f"提取到密码: {_mask(pwd)}")
            
        try:
            if pwd:
                link = re.sub(r"pwd=[^&]+", "", link)
                link = link.rstrip("?")
                out = _run(["save", link, "-p", pwd, f"/VideoAutoClip/{task_id}"]) 
            else:
                out = _run(["save", link, f"/VideoAutoClip/{task_id}"])
            
            if out and "No recent user" in out:
                _log(task_id, "WARN", "BaiduPCS用户失效，尝试重新登录")
                cookies = _env("BAIDU_PCS_COOKIES", "")
                bduss = _env("BAIDU_PCS_BDUSS", "")
                rnd_user = f"u{int(time.time())}\n"
                _run(["useradd", "--cookies", cookies, "--bduss", bduss], input_data=rnd_user)
                if pwd:
                    out = _run(["save", link, "-p", pwd, f"/VideoAutoClip/{task_id}"])
                else:
                    out = _run(["save", link, f"/VideoAutoClip/{task_id}"])
            
            if out:
                _log(task_id, "INFO", f"保存结果: {out}")
        except Exception as e:
            _log(task_id, "WARN", f"保存链接失败: {e}")
        update_progress(task_id, 5)
        # latest_dir = None
        ls_out = _run(["ls", f"/VideoAutoClip/{task_id}", "-t", "-r", "-R"])
        # try:
        #     ls_out = _run(["ls", f"/VideoAutoClip/{task_id}", "-t", "-r", "-R"])
        #     dirs = _parse_ls_dirs(ls_out)
        #     latest_dir = dirs[0] if dirs else None
        #     if latest_dir:
        #         _log(task_id, "INFO", f"选择最新目录 {latest_dir}")
        #     else:
        #         _log(task_id, "WARN", "未发现目录")
        # except Exception as e:
        #     _log(task_id, "ERROR", f"列目录失败: {e}")
        # if not latest_dir:
        #     _update_task(task_id, status="failed", finished_at=datetime.now(), error_message="未发现可用目录")
        #     update_progress(task_id, 100)
        #     return
        work_dir = _task_dir(task_id)
        names: List[str] = []
        cover_remote: Optional[str] = None
        vids: List[str] = []
        imgs: List[str] = []
        try:
            # ls2_out = _run(["ls", f"/VideoAutoClip/{latest_dir}"])
            names = sorted(_parse_ls_names(ls_out))
            imgs = [n for n in names if os.path.splitext(n)[1].lower() in [".jpg", ".jpeg", ".png", ".webp"]]
            print(f"imgs={imgs}")
            vids = [n for n in names if os.path.splitext(n)[1].lower() in [".mp4", ".mov", ".mkv", ".avi", ".m4v"]]
            #对vids进行排序如果是数字要按数字的大小排序，只对文件名排序

            vids.sort(key=lambda x: [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', x)])
            print(vids)
            
            cover_remote = imgs[0] if imgs else None
            sel = vids[:3]
            _log(task_id, "INFO", f"待下载视频 {len(sel)} 封面 {cover_remote or ''}")
            dl_targets: List[Tuple[str, str]] = []
            for nm in sel:
                dl_targets.append((f"/VideoAutoClip/{task_id}/{nm}", os.path.join(work_dir, _safe_filename(nm))))
            if cover_remote:
                dl_targets.append((f"/VideoAutoClip/{task_id}/{cover_remote}", os.path.join(work_dir, _safe_filename(cover_remote))))
        except Exception as e:
            _log(task_id, "ERROR", f"列文件失败: {e}")
            _update_task(task_id, status="failed", finished_at=datetime.now(), error_message=str(e))
            update_progress(task_id, 100)
            return
        total_bytes = 0
        for i, (remote, local) in enumerate(dl_targets):
            if os.path.exists(local):
                try:
                    sz = os.path.getsize(local)
                except Exception:
                    sz = 0
                total_bytes += sz
                _insert_file(task_id, local, "video" if local.lower().endswith((".mp4",".mov",".mkv",".avi",".m4v")) else ("cover" if local.lower().endswith((".jpg",".jpeg",".png",".webp")) else "other"), sz, i)
                continue
            try:
                _log(task_id, "INFO", f"下载 {i+1}/{len(dl_targets)} {remote}")
                _run(["download", remote, "-o", work_dir])
                base = os.path.basename(remote)
                p_local = os.path.join(work_dir, _safe_filename(base))
                p_orig = os.path.join(work_dir, base)
                if p_local != p_orig and os.path.exists(p_orig) and not os.path.exists(p_local):
                    try:
                        os.replace(p_orig, p_local)
                    except Exception:
                        pass
                path_for_insert = p_local if os.path.exists(p_local) else (p_orig if os.path.exists(p_orig) else None)
                if path_for_insert:
                    try:
                        sz = os.path.getsize(path_for_insert)
                    except Exception:
                        sz = 0
                    total_bytes += sz
                    _insert_file(task_id, path_for_insert, "video" if path_for_insert.lower().endswith((".mp4",".mov",".mkv",".avi",".m4v")) else ("cover" if path_for_insert.lower().endswith((".jpg",".jpeg",".png",".webp")) else "other"), sz, i)
            except Exception as e:
                _log(task_id, "WARN", f"下载失败 {remote} {e}")
        _log(task_id, "INFO", f"下载完成 {round(total_bytes/1024/1024,2)}MB")
        update_progress(task_id, 35)
        vids_local_names = [_resolve_local_path(work_dir, n) for n in vids]
        #获取视频的本地路径，并且存在
        vids_local = [p for p in vids_local_names if p.lower().endswith((".mp4",".mov",".mkv",".avi",".m4v")) and os.path.exists(p)]
      
        #获取封面的本地路径，并且存在
        covers_local_names=[_resolve_local_path(work_dir, n) for n in imgs]
        covers = [p for p in covers_local_names if p.lower().endswith((".jpg",".jpeg",".png",".webp")) and os.path.exists(p)]
        cover_local=covers[0] if covers else None
        print(cover_local, vids_local)
        if not vids_local:
            _update_task(task_id, status="failed", finished_at=datetime.now(), error_message="未找到视频文件")
            update_progress(task_id, 100)
            return
        merged_path = os.path.join(work_dir, "merged.mp4")
        if os.path.exists(merged_path) and not _is_video_valid(merged_path):
             print(f"已存在的合并视频无效，删除: {merged_path}")
             os.remove(merged_path)

        if not os.path.exists(merged_path):
            _concatenate_videos(vids_local, merged_path)
        update_progress(task_id, 50)
        audio_path = os.path.join(work_dir, "merged.wav")
        if not os.path.exists(audio_path):
            _export_audio(merged_path, audio_path)
        update_progress(task_id, 60)
        srt_cache_path = os.path.join(work_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".srt")
        if os.path.exists(srt_cache_path) and os.path.getsize(srt_cache_path) > 0:
            with open(srt_cache_path, "r", encoding="utf-8", errors="ignore") as f:
                srt_result = f.read()
            asr_result, asr_state = "", {}
            print(f"ASR使用缓存字幕 {srt_cache_path}")
        else:
            try:
                print(f"开始ASR识别 {audio_path}|{work_dir}")
                asr_result, srt_result, asr_state = qwen_asr_flash_async_recog(audio_path, "zh", output_dir=work_dir)
                if srt_result:
                    with open(srt_cache_path, "w", encoding="utf-8") as f:
                        f.write(srt_result)
            except Exception as e:
                print(f"ASR识别失败{e}")
                asr_result, srt_result, asr_state = "", "", {}
        if not srt_result:
            _update_task(task_id, status="failed", finished_at=datetime.now(), error_message="识别失败")
            update_progress(task_id, 100)
            return
        update_progress(task_id, 70)
        try:
            from videoautoclip.ai_short_drama_promo import create_promo_video
            model = _env("MODEL_NAME", "qwen-plus") or "qwen-plus"
            h = hashlib.md5((merged_path + srt_result).encode("utf-8")).hexdigest()
            out_path = os.path.join(OUTPUT_DIR, f"merged_video_{h}_output.mp4")
            final_path, _, message, clip_srt = create_promo_video(merged_path, srt_result, out_path, model, os.environ.get("DASHSCOPE_BASE_HTTP_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), None, './font/STHeitiMedium.ttc', 28, 1.0, 5, cover_local, None)
            _log(task_id, "INFO", message)
            _update_task(task_id, result_video_path=final_path, cover_image_path=cover_local or "")
            update_progress(task_id, 95)
        except Exception as e:
            print(f"视频生成失败{e}")
            _update_task(task_id, status="failed", finished_at=datetime.now(), error_message=str(e))
            update_progress(task_id, 100)
            return
        _update_task(task_id, status="succeeded", finished_at=datetime.now())
        update_progress(task_id, 100)
        _log(task_id, "INFO", "任务完成")
    finally:
        with _active_lock:
            _active_running.discard(task_id)

def build_ui():
    import gradio as gr
    with gr.Blocks() as app:
        with gr.Tab("设置"):
            s = get_all_settings()
            das_api = gr.Textbox(label="DASHSCOPE_API_KEY", value=_mask(s.get("DASHSCOPE_API_KEY", "")))
            pcs_cookie = gr.Textbox(label="BAIDU_PCS_COOKIES", value=_mask(s.get("BAIDU_PCS_COOKIES", "")))
            pcs_bduss = gr.Textbox(label="BAIDU_PCS_BDUSS", value=_mask(s.get("BAIDU_PCS_BDUSS", "")))
            model_name = gr.Textbox(label="MODEL_NAME", value=s.get("MODEL_NAME", "qwen-plus"))
            max_conc = gr.Textbox(label="MAX_CONCURRENT_TASKS", value=s.get("MAX_CONCURRENT_TASKS", "1"))
            base_url = gr.Textbox(label="DASHSCOPE_BASE_HTTP_API_URL", value=s.get("DASHSCOPE_BASE_HTTP_API_URL", ""))
            asr_url = gr.Textbox(label="ASR_URL", value=s.get("ASR_URL", ""))
            asr_task_url = gr.Textbox(label="ASR_TASK_URL", value=s.get("ASR_TASK_URL", ""))
            minio_endpoint = gr.Textbox(label="MINIO_ENDPOINT", value=s.get("MINIO_ENDPOINT", ""))
            minio_access_key = gr.Textbox(label="MINIO_ACCESS_KEY", value=_mask(s.get("MINIO_ACCESS_KEY", "")))
            minio_secret_key = gr.Textbox(label="MINIO_SECRET_KEY", value=_mask(s.get("MINIO_SECRET_KEY", "")))
            minio_secure = gr.Textbox(label="MINIO_SECURE", value=s.get("MINIO_SECURE", "false"))
            minio_bucket = gr.Textbox(label="MINIO_BUCKET_NAME", value=s.get("MINIO_BUCKET_NAME", "audio-files"))
            minio_base_url = gr.Textbox(label="MINIO_BASE_URL", value=s.get("MINIO_BASE_URL", ""))
            # load_btn = gr.Button("设置")
            save_btn = gr.Button("保存设置")
            status_set = gr.Textbox(label="设置状态", interactive=False)
            def _load_settings():
                s = get_all_settings()
                return s.get("DASHSCOPE_API_KEY",""), s.get("BAIDU_PCS_COOKIES",""), s.get("BAIDU_PCS_BDUSS",""), s.get("MODEL_NAME","qwen-plus"), s.get("MAX_CONCURRENT_TASKS","1"), s.get("DASHSCOPE_BASE_HTTP_API_URL",""), s.get("ASR_URL",""), s.get("ASR_TASK_URL",""), s.get("MINIO_ENDPOINT",""), _mask(s.get("MINIO_ACCESS_KEY","")), _mask(s.get("MINIO_SECRET_KEY","")), s.get("MINIO_SECURE","false"), s.get("MINIO_BUCKET_NAME","audio-files"), s.get("MINIO_BASE_URL","")
            def _save_settings(a,b,c,d,e,f,g,h,ep,ak,sk,sec,bn,bu):
                s0 = get_all_settings()
                save_settings({
                    "DASHSCOPE_API_KEY": (s0.get("DASHSCOPE_API_KEY","") if (a and "*" in a) else a),
                    "BAIDU_PCS_COOKIES": (s0.get("BAIDU_PCS_COOKIES","") if (b and "*" in b) else b),
                    "BAIDU_PCS_BDUSS": (s0.get("BAIDU_PCS_BDUSS","") if (c and "*" in c) else c),
                    "MODEL_NAME": d,
                    "MAX_CONCURRENT_TASKS": e,
                    "DASHSCOPE_BASE_HTTP_API_URL": f,
                    "ASR_URL": g,
                    "ASR_TASK_URL": h,
                    "MINIO_ENDPOINT": ep,
                    "MINIO_ACCESS_KEY": (s0.get("MINIO_ACCESS_KEY","") if (ak and "*" in ak) else ak),
                    "MINIO_SECRET_KEY": (s0.get("MINIO_SECRET_KEY","") if (sk and "*" in sk) else sk),
                    "MINIO_SECURE": sec or "false",
                    "MINIO_BUCKET_NAME": bn or "audio-files",
                    "MINIO_BASE_URL": bu or "",
                })
                return "已保存并重新加载到环境"
            # load_btn.click(_load_settings, outputs=[das_api, pcs_cookie, pcs_bduss, model_name, max_conc, base_url, asr_url, asr_task_url])
            save_btn.click(_save_settings, inputs=[das_api, pcs_cookie, pcs_bduss, model_name, max_conc, base_url, asr_url, asr_task_url, minio_endpoint, minio_access_key, minio_secret_key, minio_secure, minio_bucket, minio_base_url], outputs=[status_set])
        
        with gr.Tab("任务队列") as task_tab:
            with gr.Row():
                with gr.Column():
                    share_link = gr.Textbox(label="百度网盘链接")
                    extract_code = gr.Textbox(label="提取码(可选)")
                    add_btn = gr.Button("添加任务")
                with gr.Column():
                    tasks_df = gr.Dataframe(headers=["任务编号","标题","状态","进度","创建时间","完成时间"], datatype=["number","str","str","number","str","str"], row_count=(1,"dynamic"))
                    refresh_btn = gr.Button("刷新")
            with gr.Row():
                task_id_inp = gr.Number(label="任务ID")
                view_logs_btn = gr.Button("查看日志")
                cancel_btn = gr.Button("取消任务")
                reset_btn = gr.Button("重置任务")
                preview_btn = gr.Button("预览")
                download_btn = gr.Button("下载")
            logs_box = gr.Textbox(label="任务日志", lines=12)
            video_out = gr.Video(label="预览")
            file_out = gr.File(label="下载产物")
            def _on_df_select(evt: gr.SelectData):
                try:
                    tid = evt.row_value[0]
                    return tid, _logs(tid), gr.update(value=_preview(tid)), gr.update(value=_download(tid))
                except Exception:
                    return None, None, None, None
            def _add(link, code):
                tid = create_task({"title": link, "share_link": link, "extract_code": code})
                rows = get_tasks()
                df = [[r["id"], r.get("title",""), r.get("status",""), float(r.get("progress",0.0)), str(r.get("created_at","")), str(r.get("finished_at",""))] for r in rows]
                return f"已创建任务 {tid}", gr.update(value=""), gr.update(value=df)
            def _list():
                rows = get_tasks()
                df = [[r["id"], r.get("title",""), r.get("status",""), float(r.get("progress",0.0)), str(r.get("created_at","")), str(r.get("finished_at",""))] for r in rows]
                return gr.update(value=df)
            def _logs(tid):
                logs = get_task_logs(int(tid) if tid is not None else 0)
                return "\n".join([f"{x['timestamp']} {x['level']} {x['message']}" for x in logs])
            def _cancel(tid):
                cancel_task(int(tid))
                return f"已取消任务 {int(tid)}"
            def _reset(tid):
                if not tid:
                    return "请先选择任务"
                try:
                    _update_task(int(tid), status="queued", progress=0.0, finished_at=None, error_message="")
                    _log(int(tid), "INFO", "任务被手动重置为队列状态")
                    return f"已重置任务 {int(tid)}"
                except Exception as e:
                    return f"重置失败: {e}"
            def _preview(tid):
                conn = _connect()
                cur = conn.cursor()
                cur.execute("SELECT result_video_path FROM tasks WHERE id=?", (int(tid),))
                r = cur.fetchone()
                conn.close()
                if r and r[0]:
                    if os.path.exists(r[0]):
                        return r[0]
                    else:
                        print(f"预览文件不存在: {r[0]}")
                return None
            def _download(tid):
                conn = _connect()
                cur = conn.cursor()
                cur.execute("SELECT result_video_path FROM tasks WHERE id=?", (int(tid),))
                r = cur.fetchone()
                conn.close()
                if r and r[0]:
                    if os.path.exists(r[0]):
                        return r[0]
                    else:
                        print(f"下载文件不存在: {r[0]}")
                return None
            add_btn.click(_add, inputs=[share_link, extract_code], outputs=[logs_box, share_link, tasks_df])
            refresh_btn.click(_list, outputs=[tasks_df])
            tasks_df.select(_on_df_select, outputs=[task_id_inp, logs_box, video_out, file_out])
            view_logs_btn.click(_logs, inputs=[task_id_inp], outputs=[logs_box])
            cancel_btn.click(_cancel, inputs=[task_id_inp], outputs=[logs_box])
            reset_btn.click(_reset, inputs=[task_id_inp], outputs=[logs_box]).then(_list, outputs=[tasks_df])
            preview_btn.click(_preview, inputs=[task_id_inp], outputs=[video_out])
            download_btn.click(_download, inputs=[task_id_inp], outputs=[file_out])
            task_tab.select(_list, outputs=[tasks_df])
    return app

def start_scheduler(interval_sec: int = 5):
    def _loop():
        while True:
            try:
                run_scheduler_tick()
            except Exception:
                pass
            time.sleep(interval_sec)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()

if __name__ == "__main__":
    start_scheduler(5)
    ui = build_ui()
    ui.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7862")))
