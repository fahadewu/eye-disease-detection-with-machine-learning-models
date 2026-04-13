"""SQLite database helpers – predictions log, settings, retrain jobs."""
import sqlite3
import json
from datetime import datetime
from config import Config


def get_db():
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS predictions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        filename    TEXT,
        method      TEXT,            -- 'ensemble' | 'fallback_hf' | 'fallback_gemini'
        prediction  TEXT,
        confidence  REAL,
        all_probs   TEXT,            -- JSON list of {label, prob}
        fallback_used  INTEGER DEFAULT 0,
        fallback_note  TEXT,
        ip_address  TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS settings (
        key   TEXT PRIMARY KEY,
        value TEXT
    );

    CREATE TABLE IF NOT EXISTS retrain_jobs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        config_json TEXT,
        status      TEXT DEFAULT 'queued',  -- queued | running | done | failed
        log_text    TEXT DEFAULT '',
        started_at  TEXT,
        finished_at TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS model_overrides (
        model_key TEXT PRIMARY KEY,
        enabled   INTEGER DEFAULT 1
    );
    """)
    conn.commit()

    # Default settings
    defaults = {
        'confidence_threshold': str(Config.CONFIDENCE_THRESHOLD),
        'admin_username':       Config.ADMIN_USERNAME,
        'admin_password_hash':  Config.ADMIN_PASSWORD_HASH,
        'hf_api_key':           Config.HF_API_KEY,
        'gemini_api_key':       Config.GEMINI_API_KEY,
        'fallback_enabled':     '1',
        'fallback_priority':    'huggingface',   # 'huggingface' | 'gemini'
    }
    for k, v in defaults.items():
        c.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k, v))
        # If the row already exists but has an empty value (e.g. from a blank
        # env var on a previous deploy), overwrite it with the real default.
        if v:
            c.execute(
                "UPDATE settings SET value=? WHERE key=? AND (value IS NULL OR value='')",
                (v, k),
            )
    conn.commit()
    conn.close()


# ── Settings ───────────────────────────────────────────────────────────────────

def get_setting(key, default=None):
    conn = get_db()
    row  = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row['value'] if row else default


def set_setting(key, value):
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, str(value)))
    conn.commit()
    conn.close()


def get_all_settings():
    conn = get_db()
    rows = conn.execute("SELECT key,value FROM settings").fetchall()
    conn.close()
    return {r['key']: r['value'] for r in rows}


# ── Predictions ────────────────────────────────────────────────────────────────

def log_prediction(filename, method, prediction, confidence,
                   all_probs, fallback_used=False,
                   fallback_note='', ip_address=''):
    conn = get_db()
    conn.execute(
        """INSERT INTO predictions
           (filename,method,prediction,confidence,all_probs,
            fallback_used,fallback_note,ip_address)
           VALUES(?,?,?,?,?,?,?,?)""",
        (filename, method, prediction, confidence,
         json.dumps(all_probs), int(fallback_used),
         fallback_note, ip_address)
    )
    conn.commit()
    conn.close()


def get_predictions(limit=100, offset=0):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['all_probs'] = json.loads(d['all_probs'] or '[]')
        result.append(d)
    return result


def get_prediction_stats():
    conn = get_db()
    total   = conn.execute("SELECT COUNT(*) as n FROM predictions").fetchone()['n']
    by_class= conn.execute(
        "SELECT prediction, COUNT(*) as n FROM predictions GROUP BY prediction"
    ).fetchall()
    fb_count= conn.execute(
        "SELECT COUNT(*) as n FROM predictions WHERE fallback_used=1"
    ).fetchone()['n']
    daily   = conn.execute(
        """SELECT date(created_at) as day, COUNT(*) as n
           FROM predictions GROUP BY day ORDER BY day DESC LIMIT 14"""
    ).fetchall()
    conn.close()
    return {
        'total':    total,
        'by_class': [dict(r) for r in by_class],
        'fallback': fb_count,
        'daily':    [dict(r) for r in daily],
    }


# ── Retrain jobs ───────────────────────────────────────────────────────────────

def create_retrain_job(config_dict):
    conn = get_db()
    cur  = conn.execute(
        "INSERT INTO retrain_jobs(config_json) VALUES(?)",
        (json.dumps(config_dict),)
    )
    job_id = cur.lastrowid
    conn.commit()
    conn.close()
    return job_id


def update_retrain_job(job_id, status=None, log_append=None, finished=False):
    conn = get_db()
    if status:
        conn.execute("UPDATE retrain_jobs SET status=? WHERE id=?", (status, job_id))
    if log_append:
        conn.execute(
            "UPDATE retrain_jobs SET log_text = log_text || ? WHERE id=?",
            (log_append, job_id)
        )
    if finished:
        conn.execute(
            "UPDATE retrain_jobs SET finished_at=datetime('now') WHERE id=?",
            (job_id,)
        )
    conn.commit()
    conn.close()


def get_retrain_jobs(limit=10):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM retrain_jobs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['config'] = json.loads(d['config_json'] or '{}')
        result.append(d)
    return result


def get_retrain_job(job_id):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM retrain_jobs WHERE id=?", (job_id,)
    ).fetchone()
    conn.close()
    if row:
        d = dict(row)
        d['config'] = json.loads(d['config_json'] or '{}')
        return d
    return None


# ── Model overrides ────────────────────────────────────────────────────────────

def get_model_enabled(model_key):
    conn = get_db()
    row  = conn.execute(
        "SELECT enabled FROM model_overrides WHERE model_key=?", (model_key,)
    ).fetchone()
    conn.close()
    if row:
        return bool(row['enabled'])
    return Config.AVAILABLE_MODELS.get(model_key, {}).get('enabled', True)


def set_model_enabled(model_key, enabled: bool):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO model_overrides(model_key,enabled) VALUES(?,?)",
        (model_key, int(enabled))
    )
    conn.commit()
    conn.close()
