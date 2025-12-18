import os
import sqlite3
from datetime import datetime
from typing import Dict, Optional

DB_PATH = os.path.join(os.path.abspath("."), "data", "tasks.db")

DEFAULT_SETTINGS: Dict[str, Optional[str]] = {
    "DASHSCOPE_API_KEY": "",
    "BAIDU_PCS_COOKIES": "",
    "BAIDU_PCS_BDUSS": "",
    "MODEL_NAME": "qwen-plus",
    "MAX_CONCURRENT_TASKS": "1",
    "DASHSCOPE_BASE_HTTP_API_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "ASR_URL": "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
    "ASR_TASK_URL": "https://dashscope.aliyuncs.com/api/v1/tasks/",
    "MINIO_ENDPOINT": "",
    "MINIO_ACCESS_KEY": "",
    "MINIO_SECRET_KEY": "",
    "MINIO_SECURE": "false",
    "MINIO_BUCKET_NAME": "audio-files",
    "MINIO_BASE_URL": "",
}

def _connect():
    os.makedirs(os.path.join(os.path.abspath("."), "data"), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_settings_db():
    conn = _connect()
    cur = conn.cursor()
    for k, v in DEFAULT_SETTINGS.items():
        cur.execute("INSERT OR IGNORE INTO setting(key,value,updated_at) VALUES(?,?,?)", (k, v or "", datetime.now()))
    conn.commit()
    conn.close()

def load_settings_into_env():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT key,value FROM setting")
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        os.environ[str(r[0])] = str(r[1] or "")

def get_all_settings() -> Dict[str, str]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT key,value FROM setting ORDER BY key ASC")
    rows = cur.fetchall()
    conn.close()
    return {str(r[0]): str(r[1] or "") for r in rows}

def save_settings(values: Dict[str, Optional[str]]):
    conn = _connect()
    cur = conn.cursor()
    for k, v in values.items():
        cur.execute("INSERT OR REPLACE INTO setting(key,value,updated_at) VALUES(?,?,?)", (k, v or "", datetime.now()))
    conn.commit()
    conn.close()
    load_settings_into_env()
