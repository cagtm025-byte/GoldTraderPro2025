# tenants_auth_updated.py
# Multi-tenant auth & per-tenant DB resolver.
# Enhanced: automatically initializes tenant DB schema when create_tenant() runs.
# Place this file next to your app files. Requires: pip install passlib

import os
import sqlite3
import datetime
from typing import Tuple
from passlib.hash import pbkdf2_sha256

# Config (can be adjusted via env vars)
TENANTS_DIR = os.environ.get("TENANTS_DIR", "tenants")
AUTH_DB = os.environ.get("AUTH_DB", "auth.db")
DEFAULT_DB_FILE = os.environ.get("DEFAULT_DB_FILE", "invoices.db")

os.makedirs(TENANTS_DIR, exist_ok=True)

# ---------- Auth DB helpers ----------
def get_auth_conn():
    conn = sqlite3.connect(AUTH_DB, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_auth_db():
    conn = get_auth_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tenants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            tenant_id INTEGER,
            role TEXT DEFAULT 'user',
            created_at TEXT,
            FOREIGN KEY(tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()

# ---------- Password Hashing (pbkdf2) ----------
def hash_password(plaintext: str) -> str:
    return pbkdf2_sha256.hash(plaintext)

def verify_password(plaintext: str, password_hash: str) -> bool:
    try:
        return pbkdf2_sha256.verify(plaintext, password_hash)
    except Exception:
        return False

# ---------- Tenant DB schema initializer ----------
def init_tenant_db(db_path: str):
    """Create required tables in the tenant DB. Idempotent."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    # Basic invoices schema - keep in sync with your main app's expectations
    cur.execute("""
    CREATE TABLE IF NOT EXISTS invoices(
        invoice_no TEXT PRIMARY KEY, date TEXT, customer_name TEXT, customer_mobile TEXT,
        customer_gstin TEXT, customer_address TEXT, gst_rate REAL DEFAULT 0, gst_type TEXT,
        subtotal REAL DEFAULT 0, cgst REAL DEFAULT 0, sgst REAL DEFAULT 0, igst REAL DEFAULT 0,
        gst_total REAL DEFAULT 0, grand_total REAL DEFAULT 0, status TEXT DEFAULT 'Active', payment_status TEXT DEFAULT 'Unpaid',
        payment_mode TEXT, payment_received REAL DEFAULT 0, payment_date TEXT, cancelled_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS invoice_items(
        id INTEGER PRIMARY KEY AUTOINCREMENT, invoice_no TEXT, stock_id INTEGER, category TEXT,
        purity TEXT, hsn TEXT, item_name TEXT, qty REAL, unit TEXT, rate REAL, making REAL, amount REAL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS payments(
        id INTEGER PRIMARY KEY AUTOINCREMENT, invoice_no TEXT, customer_mobile TEXT, amount REAL,
        date TEXT, mode TEXT, note TEXT, is_advance INTEGER DEFAULT 0, created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS stocks(
        id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, purity TEXT, description TEXT,
        unit TEXT, quantity REAL, created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS stock_transactions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, stock_id INTEGER, tx_date TEXT, change REAL, reason TEXT, resulting_qty REAL, created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS customers(
        mobile TEXT PRIMARY KEY, name TEXT, gstin TEXT, address TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS company(
        id INTEGER PRIMARY KEY, name TEXT, gstin TEXT, address TEXT, logo BLOB, signature BLOB
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS advances(
        id INTEGER PRIMARY KEY AUTOINCREMENT, customer_mobile TEXT, amount REAL, remaining_amount REAL, date TEXT, mode TEXT, note TEXT, created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS advance_allocations(
        id INTEGER PRIMARY KEY AUTOINCREMENT, advance_id INTEGER, invoice_no TEXT, amount REAL, date TEXT, created_at TEXT
    )""")
    conn.commit()
    conn.close()

# inside create_tenant() after tenant_db exists
try:
    # create initial per-tenant counter file
    counter_path = os.path.join(TENANTS_DIR, f"invoice_counter_{tid}.txt")
    if not os.path.exists(counter_path):
        with open(counter_path, "w", encoding="utf-8") as cf:
            cf.write("1")
except Exception:
    pass


# ---------- Tenant & User Management ----------
def create_tenant(name: str, initialize_db: bool = True) -> Tuple[int, str]:
    """
    Create a tenant. If the name exists, append suffixes _1, _2, ...
    Returns (tenant_id, final_name).

    If initialize_db is True (default) the tenant DB schema will be created immediately.
    """
    conn = get_auth_conn(); cur = conn.cursor()
    base = (name or "").strip() or "tenant"
    candidate = base
    i = 1
    while True:
        cur.execute("SELECT 1 FROM tenants WHERE name = ?", (candidate,))
        if not cur.fetchone():
            break
        candidate = f"{base}_{i}"
        i += 1
    now = str(datetime.datetime.now())
    cur.execute("INSERT INTO tenants (name, created_at) VALUES (?, ?)", (candidate, now))
    tid = cur.lastrowid
    conn.commit(); conn.close()

    tenant_db = get_tenant_db_path(tid)
    # ensure file exists
    if not os.path.exists(tenant_db):
        open(tenant_db, "a").close()
    # initialize schema for tenant DB so it's ready to use
    if initialize_db:
        try:
            init_tenant_db(tenant_db)
        except Exception as e:
            # if init fails, attempt removal and re-raise so caller knows
            raise RuntimeError(f"Failed to initialize tenant DB ({tenant_db}): {e}")
    return tid, candidate

def create_user(username: str, password: str, tenant_id: int=None, role: str="user"):
    conn = get_auth_conn(); cur = conn.cursor()
    pw = hash_password(password)
    now = str(datetime.datetime.now())
    cur.execute("INSERT INTO users (username,password_hash,tenant_id,role,created_at) VALUES (?,?,?,?,?)",
                (username,pw,tenant_id,role,now))
    conn.commit(); conn.close()
    return True

def find_user_by_username(username: str):
    conn = get_auth_conn(); cur = conn.cursor()
    cur.execute("SELECT id,username,password_hash,tenant_id,role FROM users WHERE username=?", (username,))
    r = cur.fetchone(); conn.close()
    return r

# --- additional helpers for admin UI ---
def list_tenants():
    """Return list of tenants as tuples: (id, name, created_at)."""
    conn = get_auth_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, created_at FROM tenants ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return rows

def list_users(tenant_id: int=None):
    """
    Return list of users as tuples: (id, username, tenant_id, role, created_at).
    If tenant_id is provided, filter by that tenant.
    """
    conn = get_auth_conn()
    cur = conn.cursor()
    if tenant_id:
        cur.execute("SELECT id, username, tenant_id, role, created_at FROM users WHERE tenant_id=? ORDER BY id", (tenant_id,))
    else:
        cur.execute("SELECT id, username, tenant_id, role, created_at FROM users ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return rows

def reset_user_password(username: str, new_password: str) -> bool:
    conn = get_auth_conn()
    cur = conn.cursor()
    pw = hash_password(new_password)
    cur.execute("UPDATE users SET password_hash=?, created_at=? WHERE username= ?", (pw, str(datetime.datetime.now()), username))
    updated = cur.rowcount
    conn.commit()
    conn.close()
    return bool(updated)

def get_tenant_db_path(tenant_id: int) -> str:
    """
    Return absolute path to tenant DB file for a given tenant id.
    """
    return os.path.abspath(os.path.join(TENANTS_DIR, f"invoices_{tenant_id}.db"))

# initialize on import (idempotent)
try:
    init_auth_db()
except Exception:
    pass

# ---------- Tenant DB resolver ----------
def resolve_db_for_session(session_state: dict) -> str:
    tid = session_state.get("tenant_id")
    if tid:
        return get_tenant_db_path(tid)
    return DEFAULT_DB_FILE

def get_conn_for_session(session_state: dict):
    db_path = resolve_db_for_session(session_state)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn
