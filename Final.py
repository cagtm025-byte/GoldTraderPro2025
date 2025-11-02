
import streamlit as st
import sqlite3, os, shutil, traceback
import pandas as pd
import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import Table, TableStyle, Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from typing import Optional, Dict, Any, Sequence
import altair as alt
import base64
import json
import streamlit.components.v1 as components

# Try to import tenant auth module (optional)
try:
    import tenants_auth
    TENANTS_AUTH_AVAILABLE = True
except Exception:
    tenants_auth = None
    TENANTS_AUTH_AVAILABLE = False

# Decide whether tenant mode is enabled.
# If tenants_auth is present we enable tenant mode; else fallback to single-DB.
TENANT_ENABLED = TENANTS_AUTH_AVAILABLE

# Legacy defaults (single-DB fallback)
BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_FILE = os.path.join(DATA_DIR, "invoices.db")
DB_BACKUP_DIR = os.path.join(DATA_DIR, "db_backups")
COUNTER_FILE = os.path.join(DATA_DIR, "invoice_counter.txt")
SERVER_SAVE_DIR = os.path.join(DATA_DIR, "invoices")
os.makedirs(SERVER_SAVE_DIR, exist_ok=True)
os.makedirs(DB_BACKUP_DIR, exist_ok=True)

# Tenant-specific storage root (if tenant mode used)
TENANTS_ROOT = os.path.join(DATA_DIR, "tenants_data")
os.makedirs(TENANTS_ROOT, exist_ok=True)

# UI constants
TENANT_ENABLED = TENANT_ENABLED  # keep earlier semantics
BRAND_NAME = "GoldTrader Pro"
LEFT_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "Temple_Jewellery.jpg")

st.set_page_config(page_title="GoldTrader Pro", layout="wide", page_icon="💠")

# ---------------- Tenant-aware helpers ----------------
def current_tenant_id():
    """
    Return tenant id from session if available.
    """
    try:
        return st.session_state.get("tenant_id")
    except Exception:
        return None

def tenant_paths(tenant_id: Optional[str]):
    """
    Return paths for tenant DB, counter, invoices and backups.
    If tenant_id is falsy, return legacy single-db paths.
    """
    if not tenant_id:
        return {"db_file": DB_FILE, "counter_file": COUNTER_FILE, "save_dir": SERVER_SAVE_DIR, "backup_dir": DB_BACKUP_DIR}
    tid_str = str(tenant_id)
    tid_safe = "".join([c for c in tid_str if c.isalnum() or c in ("-", "_")]).strip()
    tdir = os.path.join(TENANTS_ROOT, tid_safe)
    os.makedirs(tdir, exist_ok=True)
    db_file = os.path.join(tdir, f"{tid_safe}.db")
    counter_file = os.path.join(tdir, f"{tid_safe}_counter.txt")
    save_dir = os.path.join(tdir, "invoices")
    backup_dir = os.path.join(tdir, "backups")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    return {"db_file": db_file, "counter_file": counter_file, "save_dir": save_dir, "backup_dir": backup_dir}

def get_conn():
    """
    Tenant-aware DB connection:
    - If tenants_auth offers get_conn_for_session, use it (keeps compatibility with super-admin style auth).
    - Else, if a tenant_id exists in st.session_state, open per-tenant sqlite file.
    - Else fallback to legacy DB_FILE.
    """
    # If tenants_auth provides a helper to get a connection for the logged-in session, use it
    try:
        if TENANTS_AUTH_AVAILABLE and hasattr(tenants_auth, "get_conn_for_session"):
            # tenants_auth should accept st.session_state or similar; we pass session_state to be flexible
            return tenants_auth.get_conn_for_session(st.session_state)
    except Exception:
        # ignore and fallback
        pass

    # fallback: use per-tenant file if tenant_id present
    tid = current_tenant_id() if TENANT_ENABLED else None
    paths = tenant_paths(tid)
    conn = sqlite3.connect(paths["db_file"], check_same_thread=False)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
    except Exception:
        pass
    return conn

# Counter helpers (tenant-aware)
def get_counter_path():
    tid = current_tenant_id() if TENANT_ENABLED else None
    return tenant_paths(tid)["counter_file"]

def get_counter():
    path = get_counter_path()
    if not os.path.exists(path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("1")
        except Exception:
            pass
        return 1
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read().strip()
            return int(data) if data else 1
    except Exception:
        return 1

def set_counter(v: int):
    path = get_counter_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(int(v)))
    except Exception:
        pass

def next_invoice_no():
    cnt = get_counter()
    inv = f"GT-{cnt:05d}"
    set_counter(cnt + 1)
    return inv

# ---------------- DB initialization (tenant-aware) ----------------
def ensure_columns(conn, table, defs):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    existing = [r[1] for r in cur.fetchall()]
    for k, v in defs.items():
        if k not in existing:
            try:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {k} {v}")
            except Exception:
                pass
    conn.commit()

def init_db():
    """
    Initialize schema on the current tenant DB (or legacy DB). Safe to call repeatedly.
    Will backup existing DB to backup directory before schema changes.
    """
    # choose path for backup based on tenant
    tid = current_tenant_id() if TENANT_ENABLED else None
    paths = tenant_paths(tid)
    db_file = paths["db_file"]
    backup_dir = paths["backup_dir"]

    # backup current DB
    try:
        if os.path.exists(db_file):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(db_file, os.path.join(backup_dir, f"{os.path.basename(db_file)}.{ts}.bak"))
    except Exception:
        pass

    conn = get_conn()
    cur = conn.cursor()
    # create base tables (as in Final.py)
    cur.execute("""CREATE TABLE IF NOT EXISTS invoices(
        invoice_no TEXT PRIMARY KEY, date TEXT, customer_name TEXT, customer_mobile TEXT,
        grand_total REAL, status TEXT, payment_status TEXT, payment_mode TEXT,
        payment_received REAL, payment_date TEXT, cancelled_at TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS invoice_items(
        id INTEGER PRIMARY KEY AUTOINCREMENT, invoice_no TEXT, category TEXT,
        purity TEXT, hsn TEXT, item_name TEXT, qty REAL, unit TEXT,
        rate REAL, making REAL, amount REAL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS payments(
        id INTEGER PRIMARY KEY AUTOINCREMENT, invoice_no TEXT, customer_mobile TEXT,
        amount REAL, date TEXT, mode TEXT, note TEXT, is_advance INTEGER DEFAULT 0,
        created_at TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS stocks(
        id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, purity TEXT,
        description TEXT, unit TEXT, quantity REAL, created_at TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS stock_transactions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, stock_id INTEGER, tx_date TEXT,
        change REAL, reason TEXT, resulting_qty REAL, created_at TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS customers(
        mobile TEXT PRIMARY KEY, name TEXT, gstin TEXT, address TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS company(
        id INTEGER PRIMARY KEY, name TEXT, gstin TEXT, address TEXT, logo BLOB, signature BLOB
    )""")
    conn.commit()

    # ensure compatibility columns (as in your original ensure_columns usage)
    ensure_columns(conn, "invoices", {
        "customer_gstin": "TEXT",
        "customer_address": "TEXT",
        "gst_rate": "REAL DEFAULT 0",
        "gst_type": "TEXT",
        "subtotal": "REAL DEFAULT 0",
        "cgst": "REAL DEFAULT 0",
        "sgst": "REAL DEFAULT 0",
        "igst": "REAL DEFAULT 0",
        "gst_total": "REAL DEFAULT 0",
        "tcs_rate": "REAL DEFAULT 0",
        "tcs_total": "REAL DEFAULT 0",
        "customer_pan": "TEXT",
        "grand_total": "REAL DEFAULT 0",
        "status": "TEXT DEFAULT 'Active'",
        "cancelled_at": "TEXT",
        "payment_status": "TEXT DEFAULT 'Unpaid'",
        "payment_mode": "TEXT",
        "payment_received": "REAL DEFAULT 0",
        "payment_date": "TEXT"
    })


    ensure_columns(conn, "invoice_items", {
        "stock_id":"INTEGER","category":"TEXT","purity":"TEXT","hsn":"TEXT","unit":"TEXT","rate":"REAL DEFAULT 0","making":"REAL DEFAULT 0","amount":"REAL DEFAULT 0"
    })
    ensure_columns(conn, "stock_transactions", {"tx_date":"TEXT","change":"REAL","reason":"TEXT","resulting_qty":"REAL","created_at":"TEXT"})
    ensure_columns(conn, "stocks", {"created_at":"TEXT"})
    ensure_columns(conn, "payments", {"invoice_no":"TEXT","customer_mobile":"TEXT","amount":"REAL DEFAULT 0","date":"TEXT","mode":"TEXT","note":"TEXT","is_advance":"INTEGER DEFAULT 0","created_at":"TEXT"})
    # advances table
    cur.execute("""CREATE TABLE IF NOT EXISTS advances (id INTEGER PRIMARY KEY AUTOINCREMENT, customer_mobile TEXT, amount REAL, remaining_amount REAL, date TEXT, mode TEXT, note TEXT, created_at TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS advance_allocations (id INTEGER PRIMARY KEY AUTOINCREMENT, advance_id INTEGER, invoice_no TEXT, amount REAL, date TEXT, created_at TEXT)""")
    conn.commit()
    conn.close()

# Make sure DB exists for current session
try:
    init_db()
except Exception:
    pass

# ---------------- Utilities ----------------
def safe_float(x, default=0.0):
    try:
        if x is None: return float(default)
        return float(x)
    except Exception:
        try:
            return float(str(x).strip() or default)
        except Exception:
            return float(default)

def table_columns(conn, table):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

def insert_dict_dynamic(conn, table, data):
    cols = table_columns(conn, table)
    keys = [k for k in data.keys() if k in cols]
    if not keys:
        raise Exception(f"No matching columns to insert for table {table}. Data keys: {list(data.keys())}, table columns: {cols}")
    placeholders = ",".join(["?"]*len(keys))
    collist = ",".join(keys)
    vals = [data[k] for k in keys]
    cur = conn.cursor()
    cur.execute(f"INSERT INTO {table} ({collist}) VALUES ({placeholders})", vals)
    conn.commit()
    return cur.lastrowid

# ---------------- CRUD helpers ----------------
# --- Helper: safe invoice dues fetch (handles missing allocations table) ---
def _get_invoice_dues(conn, customer_mobile: str) -> pd.DataFrame:
    """
    Returns DataFrame: invoice_no, date, grand_total, allocated, due
    If no allocations-like table found, allocated=0 and due=grand_total.
    """
    cur = conn.cursor()
    # Check for 'allocations' table
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='allocations'")
    row = cur.fetchone()
    alloc_table = None
    if row:
        alloc_table = "allocations"
    else:
        # Fallback: any table with 'alloc' in its name
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND lower(name) LIKE '%alloc%'")
        r = cur.fetchone()
        if r:
            alloc_table = r[0]

    # Query depending on table availability
    if alloc_table:
        sql = f"""
            SELECT i.invoice_no,
                   i.date,
                   COALESCE(i.grand_total,0) AS grand_total,
                   COALESCE(a.allocated,0) AS allocated,
                   (COALESCE(i.grand_total,0) - COALESCE(a.allocated,0)) AS due
            FROM invoices i
            LEFT JOIN (
                SELECT invoice_no, SUM(amount) AS allocated
                FROM {alloc_table}
                GROUP BY invoice_no
            ) a ON a.invoice_no = i.invoice_no
            WHERE i.customer_mobile = ? AND COALESCE(i.status,'Active') != 'Cancelled'
            ORDER BY i.date ASC
        """
    else:
        sql = """
            SELECT i.invoice_no,
                   i.date,
                   COALESCE(i.grand_total,0) AS grand_total,
                   0 AS allocated,
                   COALESCE(i.grand_total,0) AS due
            FROM invoices i
            WHERE i.customer_mobile = ? AND COALESCE(i.status,'Active') != 'Cancelled'
            ORDER BY i.date ASC
        """

    df = pd.read_sql_query(sql, conn, params=(customer_mobile,))
    df["due"] = df.get("due", df["grand_total"] - df.get("allocated", 0)).astype(float)
    return df

def fetch_customers_df():
    conn = get_conn(); df = pd.read_sql_query("SELECT * FROM customers ORDER BY name", conn); conn.close(); return df

def save_customer(mobile,name,gstin,address):
    conn = get_conn(); cur = conn.cursor(); cur.execute("INSERT OR REPLACE INTO customers (mobile,name,gstin,address) VALUES (?,?,?,?)",(mobile,name,gstin,address)); conn.commit(); conn.close()

def fetch_stocks_df():
    conn = get_conn(); df = pd.read_sql_query("SELECT * FROM stocks ORDER BY category, description", conn); conn.close(); return df

def fetch_company():
    conn = get_conn(); cur = conn.cursor(); 
    # backward compatibility: company may have been created without id
    try:
        cur.execute("SELECT name,gstin,address,logo,signature FROM company WHERE id=1")
        r = cur.fetchone()
        if not r:
            # maybe table has a single row without id indexing
            cur.execute("SELECT name,gstin,address,logo,signature FROM company LIMIT 1")
            r = cur.fetchone()
    except Exception:
        r = None
    conn.close(); return r

def save_company(name,gstin,address,logo,sig):
    conn = get_conn(); cur = conn.cursor(); 
    # ensure id column is present
    try:
        cur.execute("DELETE FROM company WHERE id=1")
        cur.execute("INSERT INTO company (id,name,gstin,address,logo,signature) VALUES (1,?,?,?,?,?)",(name,gstin,address,logo,sig))
    except Exception:
        # fallback: delete all and insert
        cur.execute("DELETE FROM company")
        cur.execute("INSERT INTO company (name,gstin,address,logo,signature) VALUES (?,?,?,?,?)",(name,gstin,address,logo,sig))
    conn.commit(); conn.close()

def load_super_overrides():
    """
    Read super_admin_overrides.json produced by super_admin.py.
    Returns dict (possibly empty) and never raises.
    """
    try:
        overrides_path = os.path.join(BASE_DIR, "super_admin_overrides.json")
        if os.path.exists(overrides_path):
            with open(overrides_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        # swallow any error and return empty dict (no overrides)
        pass
    return {}


# ---------------- Stock & Payments ----------------
def add_or_update_stock(category,purity,description,unit,change_qty,tx_date=None,reason="Manual"):
    if tx_date is None: tx_date=str(datetime.date.today())
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT id,quantity FROM stocks WHERE category=? AND purity=? AND description=?", (category,purity,description))
    r = cur.fetchone()
    if r:
        sid,existing=r[0],safe_float(r[1],0.0)
        new_qty = existing + safe_float(change_qty,0.0)
        cols = table_columns(conn,"stocks")
        if "created_at" in cols:
            cur.execute("UPDATE stocks SET quantity=?, created_at=? WHERE id=?", (new_qty,str(datetime.datetime.now()),sid))
        else:
            cur.execute("UPDATE stocks SET quantity=? WHERE id=?", (new_qty,sid))
    else:
        cols = table_columns(conn,"stocks")
        if "created_at" in cols:
            cur.execute("INSERT INTO stocks (category,purity,description,unit,quantity,created_at) VALUES (?,?,?,?,?,?)",(category,purity,description,unit,safe_float(change_qty,0.0),str(datetime.datetime.now())))
        else:
            cur.execute("INSERT INTO stocks (category,purity,description,unit,quantity) VALUES (?,?,?,?,?)",(category,purity,description,unit,safe_float(change_qty,0.0)))
        sid = cur.lastrowid; new_qty = safe_float(change_qty,0.0)
    tx = {"stock_id":sid,"tx_date":str(tx_date),"change":safe_float(change_qty),"reason":reason,"resulting_qty":safe_float(new_qty)}
    insert_dict_dynamic(conn,"stock_transactions",tx)
    conn.close()
    return sid,new_qty

def add_payment(invoice_no, amount, date=None, mode=None, note=None, is_advance=False, customer_mobile=None, conn=None):
    if date is None: date=str(datetime.date.today())
    payload={"invoice_no":invoice_no,"customer_mobile":customer_mobile,"amount":safe_float(amount,0.0),"date":str(date),"mode":mode,"note":note,"is_advance":1 if is_advance else 0,"created_at":str(datetime.datetime.now())}
    if conn is not None:
        insert_dict_dynamic(conn,"payments",payload)
    else:
        c = get_conn(); insert_dict_dynamic(c,"payments",payload); c.close()

def create_advance_note(customer_mobile, amount, date=None, mode=None, note=None):
    conn = get_conn(); cur = conn.cursor()
    if date is None: date=str(datetime.date.today())
    amt = safe_float(amount,0.0); now=str(datetime.datetime.now())
    cur.execute("INSERT INTO advances (customer_mobile, amount, remaining_amount, date, mode, note, created_at) VALUES (?,?,?,?,?,?,?)",(customer_mobile,amt,amt,str(date),mode,note,now))
    adv_id = cur.lastrowid
    add_payment(invoice_no=None, amount=amt, date=str(date), mode=mode, note=f"Advance Note #{adv_id}: {note}" if note else f"Advance Note #{adv_id}", is_advance=True, customer_mobile=customer_mobile, conn=conn)
    conn.commit(); conn.close(); return adv_id

def fetch_advances(customer_mobile=None, only_with_remaining=False, limit=500):
    conn = get_conn()
    try:
        sql = "SELECT * FROM advances"
        params = ()
        if customer_mobile and only_with_remaining:
            sql += " WHERE customer_mobile=? AND COALESCE(remaining_amount,0) > 0"
            params = (customer_mobile,)
        elif customer_mobile:
            sql += " WHERE customer_mobile=?"
            params = (customer_mobile,)
        elif only_with_remaining:
            sql += " WHERE COALESCE(remaining_amount,0) > 0"
        sql += " ORDER BY created_at DESC LIMIT ?"
        params = params + (limit,)
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return df

def allocate_advance_to_invoice(advance_id, invoice_no, amount, date=None):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id,customer_mobile,amount,remaining_amount FROM advances WHERE id=?", (advance_id,))
        adv = cur.fetchone()
        if not adv:
            raise Exception("Advance not found")
        remaining = safe_float(adv[3], 0.0)
        amt = safe_float(amount, 0.0)
        if amt <= 0:
            raise Exception("Allocation amount must be positive")
        if amt > remaining + 1e-9:
            raise Exception(f"Allocation amount ({amt}) exceeds remaining advance ({remaining})")
        if date is None:
            date = str(datetime.date.today())
        now = str(datetime.datetime.now())
        cur.execute("INSERT INTO advance_allocations (advance_id, invoice_no, amount, date, created_at) VALUES (?,?,?,?,?)", (advance_id, invoice_no, amt, str(date), now))
        new_rem = remaining - amt
        cur.execute("UPDATE advances SET remaining_amount=? WHERE id=?", (new_rem, advance_id))
        add_payment(invoice_no=invoice_no, amount=amt, date=str(date), mode=None, note=f"Applied from advance #{advance_id}", is_advance=False, customer_mobile=adv[1], conn=conn)
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    conn.close()
    return True

def fetch_allocations_for_invoice(invoice_no):
    conn = get_conn()
    try:
        df = pd.read_sql_query("SELECT a.* FROM advance_allocations a WHERE a.invoice_no=? ORDER BY a.date DESC", conn, params=(invoice_no,))
    finally:
        conn.close()
    return df

def fetch_allocations_for_advance(advance_id):
    conn = get_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM advance_allocations WHERE advance_id=? ORDER BY date DESC", conn, params=(advance_id,))
    finally:
        conn.close()
    return df

def delete_advance(advance_id, force=False):
    conn = get_conn(); cur = conn.cursor()
    try:
        allocs = pd.read_sql_query("SELECT * FROM advance_allocations WHERE advance_id=?", conn, params=(advance_id,))
        if not allocs.empty and not force:
            raise Exception("Advance has allocations. Cannot delete unless force=True.")
        if not allocs.empty and force:
            cur.execute("DELETE FROM advance_allocations WHERE advance_id=?", (advance_id,))
        cur.execute("SELECT customer_mobile, amount FROM advances WHERE id=?", (advance_id,))
        r = cur.fetchone()
        cust = r[0] if r else None
        amt = safe_float(r[1]) if r else None
        if cust is not None and amt is not None:
            cur.execute("DELETE FROM payments WHERE is_advance=1 AND customer_mobile=? AND ABS(amount - ? ) < 0.001", (cust, amt))
        cur.execute("DELETE FROM advances WHERE id=?", (advance_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    conn.close()
    return True

# ---------------- Invoice insertion & cancellation ----------------
FORCED_HSN = "7301"

def insert_invoice_and_items(invoice_no_requested, date_str, customer_mobile, customer_name, customer_gstin, customer_address, gst_rate, gst_type, items, payment_status="Unpaid", payment_mode=None, payment_received=None, payment_date=None, payment_is_advance=False, payment_note=None, customer_pan=None, tcs_rate: float = 0.0):
    """
    Inserts invoice + items. New behavior:
     - If customer's cash collections in the financial year exceed ₹200,000 and customer_pan is empty -> raise Exception (PAN required)
     - If this invoice is a cash sale and subtotal > 500,000 then enforce tcs_rate >= 1% (1.0) and compute tcs_total.
    """
    conn = get_conn(); cur = conn.cursor()

    # ---- stock availability check (unchanged) ----
    for it in items:
        sid = it.get("stock_id")
        if sid not in (None,"","None"):
            cur.execute("SELECT quantity FROM stocks WHERE id=?", (int(sid),))
            r = cur.fetchone()
            if not r: raise Exception(f"Stock id {sid} not found")
            avail = safe_float(r[0],0.0)
            if safe_float(it.get("qty",0.0)) > avail + 1e-9:
                raise Exception(f"Insufficient stock for '{it.get('item_name')}' (avail {avail})")

    # ---- totals ----
    subtotal = sum([safe_float(x.get("amount",0.0)) for x in items])
    gst_total = subtotal * (safe_float(gst_rate,0.0)/100.0)

    # ---- TCS logic: if cash sale and subtotal > 500k, ensure at least 1% ----
    tcs_rate = safe_float(tcs_rate, 0.0)
    if (payment_mode or "").lower() == "cash" and subtotal > 500000 - 1e-9:
        # enforce minimum 1% TCS for qualifying cash sales
        if tcs_rate < 1.0:
            tcs_rate = 1.0
    tcs_total = subtotal * (tcs_rate / 100.0)

    # ---- GST split ----
    if (gst_type or "").lower() == "intra-state":
        cgst = sgst = gst_total/2.0; igst = 0.0
    else:
        cgst = sgst = 0.0; igst = gst_total

    grand_total = subtotal + gst_total + tcs_total

    # ---- invoice number uniqueness logic (unchanged) ----
    cur.execute("SELECT COUNT(1) FROM invoices WHERE invoice_no=?", (invoice_no_requested,))
    r=cur.fetchone()
    if r and r[0]>0:
        candidate_num = get_counter()
        while True:
            candidate = f"GT-{candidate_num:05d}"
            cur.execute("SELECT COUNT(1) FROM invoices WHERE invoice_no=?", (candidate,))
            rr = cur.fetchone()
            if rr and rr[0]==0:
                actual_invoice_no = candidate; set_counter(candidate_num+1); break
            candidate_num += 1
    else:
        actual_invoice_no = invoice_no_requested
        try:
            requested_num = int(invoice_no_requested.split("-")[-1])
            if requested_num >= get_counter(): set_counter(requested_num+1)
        except:
            pass

    # ---- PAN mandatory check: cash collections in FY > 200k ----
    try:
        # determine financial year start/end based on invoice date_str
        inv_date = None
        try:
            inv_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            try:
                inv_date = pd.to_datetime(date_str).date()
            except Exception:
                inv_date = datetime.date.today()
        # FY start: if month >=4 then year starts April 1 of that year, else April 1 of previous year
        if inv_date.month >= 4:
            fy_start = datetime.date(inv_date.year, 4, 1)
            fy_end = datetime.date(inv_date.year + 1, 3, 31)
        else:
            fy_start = datetime.date(inv_date.year - 1, 4, 1)
            fy_end = datetime.date(inv_date.year, 3, 31)

        # Sum cash payments for this customer in FY (payments table stores mode and customer_mobile)
        cur.execute("""
            SELECT COALESCE(SUM(amount),0) FROM payments
            WHERE customer_mobile = ? AND lower(COALESCE(mode,'')) = 'cash' AND date BETWEEN ? AND ?
        """, (customer_mobile, str(fy_start), str(fy_end)))
        rsum = cur.fetchone()
        cash_collections_fy = safe_float(rsum[0], 0.0) if rsum else 0.0

        # include the present payment_received if mode is Cash (to consider the new incoming payment)
        if (payment_mode or "").lower() == "cash" and safe_float(payment_received, 0.0) > 0:
            cash_collections_fy += safe_float(payment_received, 0.0)

        if cash_collections_fy > 200000 - 1e-9 and (not customer_pan or str(customer_pan).strip() == ""):
            raise Exception("PAN is mandatory: customer cash collections in this financial year exceed ₹200,000. Please enter PAN before saving the invoice.")
    except Exception as e_pan_check:
        # rethrow if it's the PAN enforcement; for other unexpected exceptions, rethrow too
        conn.close()
        raise

    # ---- persist invoice ----
    invoice_payload = {
        "invoice_no": actual_invoice_no,
        "date": date_str,
        "customer_mobile": customer_mobile,
        "customer_name": customer_name,
        "customer_gstin": customer_gstin,
        "customer_address": customer_address,
        "customer_pan": customer_pan,
        "gst_rate": safe_float(gst_rate, 0.0),
        "gst_type": gst_type,
        "subtotal": subtotal,
        "cgst": cgst,
        "sgst": sgst,
        "igst": igst,
        "gst_total": gst_total,
        "tcs_rate": safe_float(tcs_rate, 0.0),
        "tcs_total": tcs_total,
        "grand_total": grand_total,
        "status": "Active",
        "payment_status": payment_status,
        "payment_mode": payment_mode,
        "payment_received": safe_float(payment_received,0.0),
        "payment_date": payment_date
    }
    insert_dict_dynamic(conn, "invoices", invoice_payload)

    # ---- invoice items, stock updates (unchanged) ----
    for it in items:
        item_payload = {
            "invoice_no": actual_invoice_no,
            "stock_id": it.get("stock_id"),
            "category": it.get("category"),
            "purity": it.get("purity"),
            "hsn": FORCED_HSN,
            "item_name": it.get("item_name"),
            "qty": safe_float(it.get("qty",0.0)),
            "unit": it.get("unit"),
            "rate": safe_float(it.get("rate",0.0)),
            "making": safe_float(it.get("making",0.0)),
            "amount": safe_float(it.get("amount",0.0))
        }
        insert_dict_dynamic(conn,"invoice_items", item_payload)
        if item_payload.get("stock_id") not in (None,"","None"):
            sid = int(item_payload.get("stock_id"))
            cur.execute("SELECT quantity FROM stocks WHERE id=?", (sid,))
            rr = cur.fetchone()
            if rr:
                current = safe_float(rr[0],0.0)
                new_qty = current - safe_float(item_payload["qty"],0.0)
                cols = table_columns(conn,"stocks")
                if "created_at" in cols:
                    cur.execute("UPDATE stocks SET quantity=?, created_at=? WHERE id=?", (new_qty, str(datetime.datetime.now()), sid))
                else:
                    cur.execute("UPDATE stocks SET quantity=? WHERE id=?", (new_qty, sid))
                tx_payload={"stock_id":sid,"tx_date":date_str,"change":-safe_float(item_payload["qty"],0.0),"reason":f"Sale {actual_invoice_no}","resulting_qty":safe_float(new_qty)}
                insert_dict_dynamic(conn,"stock_transactions",tx_payload)

    # ---- payments: record payment / advances ----
    if safe_float(payment_received,0.0) > 0:
        if payment_is_advance:
            conn.commit(); conn.close()
            create_advance_note(customer_mobile=customer_mobile, amount=payment_received, date=payment_date or str(datetime.date.today()), mode=payment_mode, note=payment_note)
            conn = get_conn()
        else:
            add_payment(invoice_no=actual_invoice_no, amount=payment_received, date=payment_date or str(datetime.date.today()), mode=payment_mode, note=payment_note, is_advance=False, customer_mobile=customer_mobile, conn=conn)

    conn.commit(); conn.close()
    return actual_invoice_no

def cancel_invoice(invoice_no):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT status FROM invoices WHERE invoice_no=?", (invoice_no,))
    r=cur.fetchone()
    if not r: raise Exception("Invoice not found")
    if r[0]=="Cancelled": raise Exception("Already cancelled")
    cur.execute("SELECT id,stock_id,qty FROM invoice_items WHERE invoice_no=?", (invoice_no,))
    rows = cur.fetchall()
    for item_id,stock_id,qty in rows:
        if stock_id is None: continue
        cur.execute("SELECT quantity FROM stocks WHERE id=?", (stock_id,))
        sr=cur.fetchone()
        if not sr: continue
        current = safe_float(sr[0],0.0); new_qty = current + safe_float(qty,0.0)
        cols = table_columns(conn,"stocks")
        if "created_at" in cols:
            cur.execute("UPDATE stocks SET quantity=?, created_at=? WHERE id=?", (new_qty, str(datetime.datetime.now()), stock_id))
        else:
            cur.execute("UPDATE stocks SET quantity=? WHERE id=?", (new_qty, stock_id))
        tx_payload={"stock_id":stock_id,"tx_date":str(datetime.date.today()),"change":safe_float(qty,0.0),"reason":f"Cancel {invoice_no}","resulting_qty":safe_float(new_qty)}
        insert_dict_dynamic(conn,"stock_transactions",tx_payload)
    cur.execute("UPDATE invoices SET status=?, cancelled_at=? WHERE invoice_no=?", ("Cancelled", str(datetime.datetime.now()), invoice_no))
    conn.commit(); conn.close(); return True

# ---------------- PDF generation (copied from Final.py) ----------------
FONTS_DIR = os.path.join(BASE_DIR,"fonts")
DEJAVU_REGULAR = os.path.join(FONTS_DIR,"DejaVuSans.ttf")
DEJAVU_BOLD = os.path.join(FONTS_DIR,"DejaVuSans-Bold.ttf")
_font_registered=False
try:
    if os.path.exists(DEJAVU_REGULAR):
        pdfmetrics.registerFont(TTFont("DejaVuSans",DEJAVU_REGULAR))
        if os.path.exists(DEJAVU_BOLD):
            pdfmetrics.registerFont(TTFont("DejaVuSans-Bold",DEJAVU_BOLD)); HEADER_FONT_NAME="DejaVuSans-Bold"
        else:
            HEADER_FONT_NAME="DejaVuSans"
        BASE_FONT_NAME="DejaVuSans"; _font_registered=True
    else:
        BASE_FONT_NAME="Helvetica"; HEADER_FONT_NAME="Helvetica-Bold"; _font_registered=False
except Exception:
    BASE_FONT_NAME="Helvetica"; HEADER_FONT_NAME="Helvetica-Bold"; _font_registered=False

CURRENCY_SYMBOL = "₹" if _font_registered else "Rs."

def _draw_wrapped_string(c, x, y, text, max_width, leading=12, font_name=None, font_size=9, align="left"):
    if not text: return y
    font_name = font_name or BASE_FONT_NAME
    c.setFont(font_name, font_size)
    paragraphs = str(text).splitlines()
    cur_y = y
    for para in paragraphs:
        if para.strip() == "":
            cur_y -= leading; continue
        lines = simpleSplit(para, font_name, font_size, max_width)
        for line in lines:
            if align == "left":
                c.drawString(x, cur_y, line)
            elif align == "right":
                lw = stringWidth(line, font_name, font_size)
                c.drawString(x + max_width - lw, cur_y, line)
            elif align == "center":
                lw = stringWidth(line, font_name, font_size)
                c.drawString(x + (max_width - lw) / 2.0, cur_y, line)
            else:
                c.drawString(x, cur_y, line)
            cur_y -= leading
        cur_y -= (leading * 0.15)
    return cur_y

def _fmt_val(v):
    try:
        val = float(v)
    except:
        val = 0.0
    if CURRENCY_SYMBOL == "₹":
        return f"{CURRENCY_SYMBOL}{val:,.2f}"
    else:
        return f"{CURRENCY_SYMBOL} {val:,.2f}"

def generate_invoice_pdf(invoice_no: str, invoice_row: Optional[Dict[str, Any]], items_rows: Optional[Sequence[Dict[str, Any]]], company_row: Optional[tuple] = None, terms_text: Optional[str] = None) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    left = 18 * mm; right = width - 18 * mm; top = height - 20 * mm; usable_w = right - left

    comp_name = company_row[0] if company_row else "GoldTrader Pro"
    comp_gstin = company_row[1] if company_row else ""
    comp_addr = company_row[2] if company_row else ""
    logo_bytes = company_row[3] if company_row and len(company_row)>3 else None
    sig_bytes = company_row[4] if company_row and len(company_row)>4 else None

    c.setFont(HEADER_FONT_NAME, 18); c.setFillColor(colors.HexColor("#082b3a"))
    if logo_bytes:
        try:
            img = Image(BytesIO(logo_bytes)); img.drawHeight = 18*mm; img.drawWidth = 36*mm
            img.wrapOn(c, 36*mm, 18*mm); img.drawOn(c, left, top - 18*mm)
        except:
            c.drawString(left, top - 10, comp_name)
    else:
        c.drawString(left, top - 10, comp_name)
    c.setFont(BASE_FONT_NAME, 8); _draw_wrapped_string(c, left + usable_w*0.55, top - 16, comp_addr, max_width=usable_w*0.45, leading=10, font_size=8, font_name=BASE_FONT_NAME, align="right")
    if comp_gstin: c.setFont(BASE_FONT_NAME, 9); c.drawRightString(right, top - 26, f"GSTIN: {comp_gstin}")

    c.setFont(HEADER_FONT_NAME, 14); c.setFillColor(colors.HexColor("#082b3a"))
    c.drawCentredString(left + usable_w/2.0, top - 76, "TAX INVOICE")
    meta_y = top - 86
    c.setFont(BASE_FONT_NAME, 9); c.setFillColor(colors.black)
    c.drawString(left, meta_y, f"Invoice No: {invoice_no}")
    if invoice_row and invoice_row.get("date"): c.drawRightString(right, meta_y, f"Date: {invoice_row.get('date')}")

    by = meta_y - 20; c.setFont(HEADER_FONT_NAME, 10); c.drawString(left, by, "Bill To:")
    by -= 12; c.setFont(BASE_FONT_NAME, 9)
    if invoice_row:
        c.drawString(left+6, by, invoice_row.get("customer_name","")); by -= 12
        c.drawString(left+6, by, f"Mob: {invoice_row.get('customer_mobile','')}"); by -= 12
        if invoice_row.get("customer_gstin"): c.drawString(left+6, by, f"GSTIN: {invoice_row.get('customer_gstin')}"); by -= 12
        if invoice_row.get("customer_address"): by = _draw_wrapped_string(c, left+6, by, invoice_row.get("customer_address"), max_width=usable_w*0.6, leading=10, font_size=8)

    table_top = by - 8
    header = ["SNo","Category/Purity","Description","HSN","Qty","Unit","Rate","Making","Amount"]
    rows = [header]
    items_rows = list(items_rows or [])
    for i,it in enumerate(items_rows, start=1):
        rows.append([
            str(i),
            f"{it.get('category','')}/{it.get('purity','')}",
            it.get('item_name',''),
            FORCED_HSN,
            f"{safe_float(it.get('qty',0.0)):g}",
            it.get('unit',''),
            f"{safe_float(it.get('rate',0.0)):,.2f}",
            f"{safe_float(it.get('making',0.0)):,.2f}",
            f"{safe_float(it.get('amount',0.0)):,.2f}"
        ])

    col_widths = [28, usable_w*0.22, usable_w*0.18, 40, 32, 32, usable_w*0.14, usable_w*0.10, usable_w*0.18]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        ('FONT', (0,0), (-1,0), HEADER_FONT_NAME),
        ('FONT', (0,1), (-1,-1), BASE_FONT_NAME),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#c79a2e")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor("#e0e0e0"))
    ])
    tbl.setStyle(style)
    w_tbl, h_tbl = tbl.wrapOn(c, usable_w, 400)
    tbl.drawOn(c, left, table_top - h_tbl)

    subtotal = float(invoice_row.get("subtotal",0) if invoice_row else sum(safe_float(it.get("amount",0.0)) for it in items_rows))
    gst_total = float(invoice_row.get("gst_total",0) if invoice_row else (subtotal * (safe_float(invoice_row.get("gst_rate",0.0))/100.0) if invoice_row else 0.0))
    tcs_total = float(invoice_row.get("tcs_total", 0.0) if invoice_row else 0.0)
    if invoice_row and (invoice_row.get("gst_type") or "").lower()=="intra-state":
        cgst = gst_total/2.0; sgst = gst_total/2.0; igst = 0.0
    else:
        cgst = 0.0; sgst = 0.0; igst = gst_total
    grand_total = float(invoice_row.get("grand_total", subtotal + gst_total + tcs_total) if invoice_row else subtotal + gst_total + tcs_total)

    totals_x = left + usable_w*0.45
    ty = (table_top - h_tbl) - 20
    c.setFont(BASE_FONT_NAME, 9); c.setFillColor(colors.HexColor("#4d4d4d"))
    c.drawRightString(right - 8, ty, f"Subtotal: {_fmt_val(subtotal)}"); ty -= 14
    if cgst and sgst:
        c.drawRightString(right - 8, ty, f"CGST: {_fmt_val(cgst)}"); ty -= 14
        c.drawRightString(right - 8, ty, f"SGST: {_fmt_val(sgst)}"); ty -= 16
    else:
        c.drawRightString(right - 8, ty, f"IGST: {_fmt_val(igst)}"); ty -= 18

    # --- TCS (if present) ---
    if tcs_total and tcs_total > 0:
        tcs_rate_display = f" ({safe_float(invoice_row.get('tcs_rate',0.0)):.2f}%)" if invoice_row else ""
        c.drawRightString(right - 8, ty, f"TCS{tcs_rate_display}: {_fmt_val(tcs_total)}"); ty -= 16

    c.setFont(HEADER_FONT_NAME, 12); c.setFillColor(colors.HexColor("#c79a2e"))
    c.drawRightString(right - 8, ty, f"Grand Total: {_fmt_val(grand_total)}")

    p_line_y = ty - 18
    payment_status = invoice_row.get("payment_status","") if invoice_row else ""
    payment_mode = invoice_row.get("payment_mode","") if invoice_row else ""
    payment_received = float(invoice_row.get("payment_received",0) or 0) if invoice_row else 0
    payment_date = invoice_row.get("payment_date","") if invoice_row else ""
    p_line = f"Payment Status: {payment_status}"
    if payment_mode: p_line += f" | Mode: {payment_mode}"
    if payment_received > 0: p_line += f" | Received: {_fmt_val(payment_received)} on {payment_date}"
    c.setFont(BASE_FONT_NAME, 9); c.setFillColor(colors.black)
    c.drawString(left, p_line_y, p_line)

    t_x = left; t_y = p_line_y - 36
    c.setFont(HEADER_FONT_NAME, 9); c.setFillColor(colors.HexColor("#082b3a"))
    c.drawString(t_x, t_y, "Terms & Conditions")
    c.setFont(BASE_FONT_NAME, 8); c.setFillColor(colors.black)
    default_terms = "1. Goods once sold will not be taken back unless agreed in writing. 2. Verify items on receipt. Claims after 7 days may not be accepted."
    _draw_wrapped_string(c, t_x, t_y - 12, terms_text if terms_text else default_terms, max_width=usable_w*0.7, leading=10, font_size=8)

    c.line(left + 6, left + 48, left + 120, left + 48)
    c.drawString(left + 6, left + 34, "Customer Signature")
    auth_x = left + usable_w - 120
    if sig_bytes:
        try:
            s_img = Image(BytesIO(sig_bytes))
            s_img.drawHeight = 18; s_img.drawWidth = 100
            s_img.wrapOn(c, 100, 18); s_img.drawOn(c, auth_x + 6, left + 28)
        except:
            pass
    c.line(auth_x + 6, left + 48, auth_x + 120, left + 48)
    c.drawString(auth_x + 6, left + 34, "Authorised Signatory")

    c.setFont("Helvetica-Oblique", 8); c.setFillColor(colors.HexColor("#6b7280"))
    c.drawCentredString(left + usable_w/2.0, 10*mm, "Thank you for your business! This is a computer-generated invoice.")

    c.showPage(); c.save(); buffer.seek(0)
    return buffer

# ---------------- Read helpers ----------------
def read_invoice_from_db(invoice_no):
    conn = get_conn()
    inv = pd.read_sql_query("SELECT * FROM invoices WHERE invoice_no=?", conn, params=(invoice_no,))
    items = pd.read_sql_query("SELECT * FROM invoice_items WHERE invoice_no=?", conn, params=(invoice_no,))
    conn.close()
    if inv.empty: return None, None
    return inv.iloc[0].to_dict(), items.to_dict(orient="records")

# ---------------- UI state defaults ----------------
if "invoice_items" not in st.session_state: st.session_state["invoice_items"] = []
if "draft_invoice_no" not in st.session_state: st.session_state["draft_invoice_no"] = None
if "page" not in st.session_state: st.session_state["page"] = "Dashboard"

# ---------------- Login page (preserve your layout) ----------------
if TENANT_ENABLED:
    if "username" not in st.session_state or not st.session_state.get("username"):
        st.set_page_config(page_title=f"{BRAND_NAME} — Sign in", layout="wide")

        # CSS Styling
        st.markdown(
            """
            <style>
            .header-card {
                display: flex;
                align-items: center;
                margin-bottom: 20px;
            }
            .logo-circle {
                background: #facc15; /* gold */
                color: black;
                font-weight: 700;
                font-size: 24px;
                width: 48px;
                height: 48px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 14px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            }
            .title-block {
                display: flex;
                flex-direction: column;
            }
            .title-main {
                font-size: 24px;
                font-weight: 700;
            }
            .title-sub {
                font-size: 13px;
                color: #6b7280;
            }
            .brand-sub {
                color: #6b7280; /* gray-500 */
                margin-bottom: 18px;
            }
            .small-note { font-size: 12px; color: #9ca3af; }
            .login-box { padding: 8px 0 4px 0; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        left_col, right_col = st.columns([0.8, 1])
        with left_col:
            if os.path.exists(LEFT_IMAGE_PATH):
                st.image(LEFT_IMAGE_PATH, use_container_width=True)
            else:
                st.image("https://via.placeholder.com/600x800.png?text=GoldTrader+Pro+Login", use_container_width=True)

        with right_col:
            st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='login-box'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='header-card'>"
                "<div class='logo-circle'>GT</div>"
                "<div class='title-block'>"
                "<div class='title-main'>GoldTrader Pro</div>"
                "<div class='title-sub'>Invoicing & Inventory — simple, secure, professional</div>"
                "</div>"
                "</div>",
                unsafe_allow_html=True
            )

            auth_mode = st.selectbox("Mode", ["Login", "Admin: Create Tenant & Admin User"])

            if auth_mode == "Login":
                with st.form(key="login_form"):
                    username = st.text_input("Username", key="login_usr")
                    password = st.text_input("Password", type="password", key="login_pwd")
                    submitted = st.form_submit_button("Login")

                if submitted:
                    try:
                        if TENANTS_AUTH_AVAILABLE:
                            u = tenants_auth.find_user_by_username(username)
                        else:
                            u = None
                    except Exception as e:
                        st.error(f"Authentication backend error: {e}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.stop()

                    if not u:
                        st.error("Invalid username or password")
                    else:
                        # unpack user record
                        uid, uname, phash, tenant_id, role = u

                        # verify password first
                        if TENANTS_AUTH_AVAILABLE and tenants_auth.verify_password(password, phash):
                            # Check super-admin overrides to see if this tenant is enabled
                            try:
                                overrides = load_super_overrides()
                                # tenant_id may be int or string or None. Normalize to string for lookup.
                                tid_key = str(tenant_id) if tenant_id is not None else ""
                                tenant_meta = overrides.get(tid_key, {}) if tid_key else {}
                                tenant_enabled = tenant_meta.get("enabled", True)
                            except Exception:
                                # If anything goes wrong reading overrides, default to allowing login
                                tenant_enabled = True

                            if not tenant_enabled:
                                st.error("This tenant has been disabled by the administrator. Contact your super-admin.")
                            else:
                                # successful login: populate session and init tenant DB
                                st.session_state["username"] = uname
                                st.session_state["user_id"] = uid
                                st.session_state["tenant_id"] = tenant_id
                                st.session_state["role"] = role
                                try:
                                    init_db()
                                except Exception as e_init:
                                    st.warning(f"Tenant DB init warning: {e_init}")
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    pass
                        else:
                            st.error("Invalid username or password")

            else:
                with st.form(key="create_tenant_form"):
                    admin_key = st.text_input("Server passphrase", type="password", key="admin_key")
                    new_tenant_name = st.text_input("Tenant name", key="new_tenant_name")
                    admin_user = st.text_input("Admin username", key="new_admin_username")
                    admin_pwd = st.text_input("Admin password", type="password", key="new_admin_password")
                    create_sub = st.form_submit_button("Create tenant and admin")

                if create_sub:
                    SERVER_BOOT_PASSPHRASE = os.environ.get("TENANT_BOOT_PASSPHRASE", "@Gsf025@")
                    if admin_key != SERVER_BOOT_PASSPHRASE:
                        st.error("Invalid server passphrase")
                    elif not new_tenant_name or not admin_user or not admin_pwd:
                        st.error("Tenant & admin required")
                    else:
                        try:
                            if not TENANTS_AUTH_AVAILABLE:
                                raise Exception("tenants_auth module not available on this server.")
                            tid, final_name = tenants_auth.create_tenant(new_tenant_name)
                            tenants_auth.create_user(admin_user, admin_pwd, tenant_id=tid, role="admin")
                            st.success(f"Tenant created: '{final_name}' (id={tid}). Admin user created.")
                        except Exception as e:
                            st.error(f"Could not create tenant: {e}")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='small-note'>Trouble signing in? Contact your administrator.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.stop()

# If logged-in or tenant-mode disabled, continue with the app.
if "username" in st.session_state and st.session_state.get("username"):
    st.sidebar.success(f"Signed in as: {st.session_state.get('username')}")

# Ensure DB initialized for current tenant/session
try:
    init_db()
except Exception:
    pass

import streamlit as st
from urllib.parse import quote, unquote

# ---------------- Read query params (new API) ----------------
qp = st.query_params or {}
page_param = None
action_param = None

# Handle query params safely
if "page" in qp and qp["page"]:
    raw = qp.get("page")
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if len(raw) > 0 else ""
    try:
        page_param = unquote(raw)
    except Exception:
        page_param = raw

if "action" in qp and qp["action"]:
    raw_a = qp.get("action")
    if isinstance(raw_a, (list, tuple)):
        raw_a = raw_a[0] if len(raw_a) > 0 else ""
    try:
        action_param = unquote(raw_a)
    except Exception:
        action_param = raw_a


# Helper to update query params (new API)
def set_query_page(page=None, action=None):
    params = {}
    if page:
        params["page"] = page
    if action:
        params["action"] = action
    try:
        if params:
            st.set_query_params(**params)
        else:
            st.set_query_params()
    except Exception:
        try:
            st.experimental_set_query_params(**params)
        except Exception:
            pass


# Handle logout action
if action_param and str(action_param).lower() == "logout":
    for k in ["username", "user_id", "tenant_id", "role", "page", "invoice_items", "draft_invoice_no"]:
        if k in st.session_state:
            del st.session_state[k]
    set_query_page()
    try:
        st.rerun()
    except Exception:
        pass

# If a page param arrived, honor it
if page_param:
    st.session_state["page"] = page_param

# Ensure defaults
if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

import streamlit as st
from urllib.parse import quote

# ---------------- CSS (Golden 3D Pill Buttons + Spacing) ----------------
st.markdown(
    """
    <style>
    /* Sidebar base container */
    div[data-testid="stSidebar"] > div:first-child {
        padding: 24px 18px;
        background: linear-gradient(180deg, #081422 0%, #0c2438 100%);
        border-radius: 14px;
        box-shadow: 0 14px 38px rgba(2,6,23,0.6);
    }

    /* Header section */
    .sidebar-header {
        display:flex; align-items:center; gap:12px;
        padding-bottom:12px; margin-bottom:18px;
        border-bottom:1px solid rgba(255,255,255,0.05);
    }
    .gt-logo {
        width:56px; height:56px; border-radius:14px;
        background: linear-gradient(135deg, #facc15, #f59e0b);
        display:flex; align-items:center; justify-content:center;
        font-weight:900; font-size:22px; color:#082032;
        box-shadow: 0 8px 22px rgba(245,158,11,0.45);
    }
    .title-main { font-size:18px; color:#FDF6E3; font-weight:800; }
    .title-sub { font-size:12px; color:rgba(253,246,227,0.75); }

    /* Unified button style */
    .menu-container, .profile-dropdown {
        display:flex;
        flex-direction:column;
        gap:10px;
    }

    .menu-link, .profile-action {
        text-decoration:none;
        display:block;
    }

    .menu-item, .profile-item {
        display:flex;
        align-items:center;
        gap:12px;
        padding:14px 16px;
        border-radius:12px;
        background: linear-gradient(90deg,#0a1e36,#0c2a46);
        color:#EAF2F8;
        font-weight:800;
        box-shadow: 0 10px 24px rgba(2,6,23,0.45);
        border-left: 6px solid #f59e0b;
        transition: all 0.12s ease-in-out;
    }

    .menu-item:hover, .profile-item:hover {
        transform: translateY(-3px);
        background: linear-gradient(90deg,#0b2542,#0d3155);
        box-shadow: 0 16px 36px rgba(2,6,23,0.65);
    }

    .menu-item.selected {
        background: linear-gradient(90deg,#0b294b,#0d3a6b);
        color:#FFF;
        box-shadow: 0 18px 44px rgba(245,158,11,0.45);
    }

    .menu-emoji, .profile-emoji {
        font-size:18px; width:28px; text-align:center;
    }
    .menu-label, .profile-label {
        font-size:15px;
    }

    /* Divider line + spacing between main menu and profile actions */
    .menu-divider {
        height:1px;
        background: rgba(255,255,255,0.08);
        margin:18px 0 12px 0;
    }

    /* Footer */
    .sidebar-footer {
        margin-top:20px;
        color:rgba(255,255,255,0.55);
        font-size:12px;
        text-align:left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar Header ----------------
st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <div class="gt-logo">GT</div>
        <div style="display:flex;flex-direction:column;">
            <div class="title-main">GoldTrader Pro</div>
            <div class="title-sub">Invoicing • Inventory • CRM</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

import streamlit as st
from urllib.parse import quote

# ---------------- Menu Buttons ----------------
menu_options = [
    ("📊", "Dashboard"),
    ("🧾", "Create Invoice"),
    ("🛒", "Report & Register"),
    ("💳", "Payments Ledger"),
    ("📦", "Stock Master"),
    ("🏢","Company Settings"),
]

menu_html = "<div class='menu-container'>"
for emoji, label in menu_options:
    selected = "selected" if st.session_state.get("page") == label else ""
    href = f"?page={quote(label)}"
    
    # Button using st.sidebar.button for functionality, while updating session state.
    if st.sidebar.button(f"{emoji} {label}", key=f"btn_{label}"):
        st.session_state["page"] = label
        
        
    # Reliable Logout button (handled here, not as a 'page')
if st.sidebar.button("🔓 Logout", key="btn_logout"):
    # keys to remove on logout
    for k in ["username","user_id","tenant_id","role", "page", "invoice_items", "draft_invoice_no"]:
        if k in st.session_state:
            del st.session_state[k]
    # try a safe rerun
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass

st.sidebar.markdown("<div style='margin-top:12px; font-size:14px; color:#6b7280;'>@ hosted at www.goldtraderpro.in</div>", unsafe_allow_html=True)

# ---------------- Header (Main Page) ----------------
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:16px;padding:14px 20px;border-radius:14px;
                background:linear-gradient(90deg,#0A1F44,#0C2438);box-shadow:0 8px 22px rgba(2,6,23,0.45);margin-bottom:18px;">
        <div style="width:58px;height:58px;border-radius:14px;display:flex;align-items:center;justify-content:center;
                    font-size:24px;background:linear-gradient(135deg,#facc15,#f59e0b);font-weight:900;color:#082032;
                    box-shadow:0 8px 20px rgba(245,158,11,0.45);">GT</div>
        <div style="display:flex;flex-direction:column;">
            <div style="font-size:19px;color:#FDF6E3;font-weight:800;">GoldTrader Pro</div>
            <div style="font-size:13px;color:rgba(253,246,227,0.85);">Invoicing & Inventory — simple, secure, professional</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Page Routing ----------------
page = st.session_state.get("page", "Dashboard")

if page == "Dashboard":
    pass
elif page == "Create Invoice":
    pass
elif page == "Report & Register":
    pass
elif page == "Payments Ledger":
    pass
elif page == "Stock Master":
    pass
elif page == "Company Settings":
    pass
else:
    st.write(f"You are viewing: {page}")


# ---------- Dashboard page block (Category | QTY only) ----------
if page == "Dashboard":
    import numpy as np
    from functools import reduce

    def _company_name_or_default():
        company = fetch_company()
        return company[0] if company and company[0] else "GoldTrader Pro"

    def _load_data():
        conn = get_conn()
        try:
            df_inv = pd.read_sql_query(
                "SELECT invoice_no, date, COALESCE(grand_total,0) AS grand_total FROM invoices", conn)
            df_pay = pd.read_sql_query(
                "SELECT invoice_no, COALESCE(amount,0) AS amount, date, COALESCE(is_advance,0) AS is_advance FROM payments", conn)
            stocks_df = pd.read_sql_query("SELECT * FROM stocks", conn)
        finally:
            try: conn.close()
            except Exception: pass
        return df_inv, df_pay, stocks_df

    def _to_date_safe(df, col):
        if col not in df.columns or df.empty: return df
        try: df[col] = pd.to_datetime(df[col]).dt.date
        except Exception: pass
        return df

    def _sum_in_range(df, date_col, start, end, val_col):
        if df.empty: return 0.0
        _to_date_safe(df, date_col)
        sel = df[(df[date_col] >= start) & (df[date_col] <= end)]
        return float(sel[val_col].sum()) if not sel.empty else 0.0

    def _responsive_kpi_cards(sales_today, cash_on_hand, advances_total, total_outstanding):
        st.markdown("""
        <style>
        .kpi-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                   gap: 16px; margin-bottom: 20px;}
        .kpi-card {padding:14px; border-radius:12px;}
        .kpi-title {font-size:13px; font-weight:700;}
        .kpi-value {font-size:24px; font-weight:800;}
        </style>
        """, unsafe_allow_html=True)
        cards = [
            ("#fff7ed,#ffedd5", "#92400e", "Sales (Today)", sales_today),
            ("#f0f9ff,#e0f2fe", "#0f172a", "Cash on Hand (Snapshot)", cash_on_hand),
            ("#ecfccb,#bbf7d0", "#065f46", "Advance", advances_total),
            ("#fce7f3,#e9d5ff", "#5b21b6", "Total Outstanding", total_outstanding)
        ]
        html = "<div class='kpi-grid'>"
        for grad, color, title, value in cards:
            html += (f"<div class='kpi-card' style='background:linear-gradient(135deg,{grad});'>"
                     f"<div class='kpi-title' style='color:{color}'>{title}</div>"
                     f"<div class='kpi-value' style='color:{color}'>₹{value:,.2f}</div></div>")
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    def _prepare_daily_sales(df_inv, start_date, end_date):
        df = df_inv.copy() if not df_inv.empty else pd.DataFrame(columns=["date","grand_total"])
        _to_date_safe(df,"date")
        df_daily = (df[(df["date"] >= start_date)&(df["date"] <= end_date)]
            .groupby("date",as_index=False)
            .agg(total_amount=pd.NamedAgg("grand_total","sum")))
        idx = pd.date_range(start=start_date,end=end_date)
        df_daily_all = pd.DataFrame({"date":idx.date})
        df_daily = pd.merge(df_daily_all,df_daily,how="left",on="date").fillna(0)
        return df_daily

    def _render_sales_chart(df_daily):
        if df_daily["total_amount"].sum()==0:
            st.info("No sales recorded in the last 7 days."); return
        chart = (alt.Chart(df_daily).mark_area().encode(
            x=alt.X("date:T",title="Date"),
            y=alt.Y("total_amount:Q",title="Sales (₹)"),
            tooltip=[alt.Tooltip("date:T",title="Date"),
                     alt.Tooltip("total_amount:Q",title="Sales",format=",.2f")]))
        st.altair_chart(chart.properties(height=300),use_container_width=True)

    def _render_stock_overview(stocks_df):
        if stocks_df.empty: st.info("No stock records found."); return
        cols_lower={c.lower():c for c in stocks_df.columns}
        cat_col=cols_lower.get("category") or cols_lower.get("group") or cols_lower.get("grp")
        if not cat_col: cat_col=list(stocks_df.columns)[0]
        qty_candidates=["closing_qty","available_qty","quantity","qty","stock"]
        qty_col=next((cols_lower[c] for c in qty_candidates if c in cols_lower),None)
        if qty_col is None:
            for c in stocks_df.columns:
                if pd.api.types.is_numeric_dtype(stocks_df[c]): qty_col=c; break
        if qty_col is None: qty_col=stocks_df.columns[0]
        stocks_df[qty_col]=pd.to_numeric(stocks_df[qty_col],errors="coerce").fillna(0.0)
        summary=(stocks_df.groupby(cat_col).agg(QTY=(qty_col,"sum"))
            .reset_index().rename(columns={cat_col:"Category"}))
        summary["QTY"]=summary["QTY"].astype(float).round(2)
        display_df=summary.copy(); display_df["QTY"]=display_df["QTY"].map(lambda x:f"{x:,.2f}")
        st.table(display_df[["Category","QTY"]].sort_values("QTY",ascending=False).reset_index(drop=True))

    company_name=_company_name_or_default()
    st.markdown(f"## Welcome! — {company_name}")
    st.markdown("---"); st.header("Sales & Finance Snapshot")

    df_inv,df_pay,stocks_df=_load_data()
    today=datetime.date.today(); last_7=today-datetime.timedelta(days=6)
    sales_today=_sum_in_range(df_inv,"date",today,today,"grand_total")
    cash_on_hand=float(df_pay["amount"].sum()) if not df_pay.empty else 0.0
    advances_total=float(df_pay.loc[df_pay["is_advance"]==1,"amount"].sum()) if not df_pay.empty else 0.0

    conn=get_conn()
    total_outstanding_df=pd.read_sql_query("""
        SELECT inv.customer_mobile, inv.customer_name, COALESCE(inv.grand_total,0) AS grand_total,
               COALESCE(applied.applied_sum,0) AS applied
        FROM invoices inv
        LEFT JOIN (
            SELECT invoice_no, COALESCE(SUM(amount),0) AS applied_sum
            FROM payments WHERE COALESCE(is_advance,0)=0 GROUP BY invoice_no
        ) applied ON applied.invoice_no=inv.invoice_no
        WHERE COALESCE(inv.status,'Active')!='Cancelled'""",conn)
    try: conn.close()
    except Exception: pass
    total_outstanding=float((total_outstanding_df["grand_total"]-total_outstanding_df["applied"]).sum()) if not total_outstanding_df.empty else 0.0

    _responsive_kpi_cards(sales_today,cash_on_hand,advances_total,total_outstanding)
    st.markdown("---")
    left_col,right_col=st.columns([1,1],gap="large")
    with left_col:
        st.subheader("Sales — Daily Performance (7 days)")
        if df_inv.empty: st.info("No sales data.")
        else:
            _to_date_safe(df_inv,"date")
            df_daily=_prepare_daily_sales(df_inv,last_7,today)
            if df_daily["total_amount"].sum()==0: st.info("No sales recorded in the last 7 days.")
            else: _render_sales_chart(df_daily)
    with right_col:
        st.subheader("Stock Overview")
        _render_stock_overview(stocks_df)
# ---------- end of Dashboard block ----------


# --- Create Invoice (Compact + Inline-editable Items + PAN & TCS & Polish Charge) ----
elif page == "Create Invoice":
    import base64
    import json
    import traceback
    import datetime
    import streamlit as st
    import streamlit.components.v1 as components
    from io import BytesIO
    import pandas as pd
    import re

    # ---------- Theme / CSS (Charcoal + Royal Blue) ----------
    st.markdown(
        """
    <style>
    body, [data-testid="stAppViewContainer"] {
        background-color: #1C1C1C !important;
        color: #E5E5E5 !important;
    }
    .section-header {
        background: linear-gradient(90deg, #0A1F44 0%, #1C1C1C 100%);
        color: #E5E5E5;
        padding: 8px 14px;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .crm-card { background-color:#0A1F44; padding:12px; border-radius:8px; }
    .stButton > button { background-color: #0A1F44 !important; color: #fff !important; border-radius:6px !important; }
    .stButton > button:hover { background-color:#142952 !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ---------- Helpers ----------
    def safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    def auto_download_pdf(pdf_bytes, filename="invoice.pdf"):
        """Auto-download (attempt via JS)"""
        try:
            data_bytes = pdf_bytes.getvalue() if hasattr(pdf_bytes, "getvalue") else bytes(pdf_bytes)
            b64 = base64.b64encode(data_bytes).decode()
            html = f"""
            <a id="dl_anchor" style="display:none" href="data:application/pdf;base64,{b64}" download="{filename}"></a>
            <script>document.getElementById('dl_anchor').click();</script>
            """
            components.html(html, height=0)
        except Exception as e:
            st.warning(f"Auto-download failed: {e}")

    # ---------- Stock Check ----------
    stocks_df = pd.DataFrame()
    try:
        stocks_df = fetch_stocks_df() if "fetch_stocks_df" in globals() else pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load stocks: {e}")
        stocks_df = pd.DataFrame()

    if stocks_df.empty:
        st.error("❌ Cannot create invoice — Stock Master is empty.")
        st.stop()

    # ---------- Draft controls ----------
    with st.expander("🧾 Start or Manage Invoice", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            if not st.session_state.get("draft_invoice_no"):
                if st.button("Start New Invoice", use_container_width=True, key="start_new_invoice"):
                    try:
                        st.session_state["draft_invoice_no"] = next_invoice_no()
                    except Exception:
                        st.session_state["draft_invoice_no"] = f"DRAFT-{int(datetime.datetime.now().timestamp())}"
                    st.session_state["invoice_items"] = []
                    st.success(f"Started draft: {st.session_state['draft_invoice_no']}")
            else:
                st.info(f"Active Draft: {st.session_state['draft_invoice_no']}")
                if st.button("Clear Draft", use_container_width=True, key="clear_draft"):
                    st.session_state["draft_invoice_no"] = None
                    st.session_state["invoice_items"] = []
        with col2:
            # keep a stable session key so we can read later
            inv_date = st.date_input("Invoice Date", value=datetime.date.today(), key="inv_date_create")

    if not st.session_state.get("draft_invoice_no"):
        st.stop()

    invoice_no = st.session_state["draft_invoice_no"]
    st.markdown(
        f"<div class='section-header'>Invoice No: {invoice_no} | Date: {inv_date.strftime('%d-%m-%Y')}</div>",
        unsafe_allow_html=True,
    )

    # ---------- Customer section (PAN included + STATE mandatory dropdown) ----------
    with st.expander("👤 Customer Details", expanded=True):
        try:
            customers_df = fetch_customers_df() if "fetch_customers_df" in globals() else pd.DataFrame()
        except Exception as e:
            st.warning(f"Could not load customers: {e}")
            customers_df = pd.DataFrame()

        cust_options = ["--New Customer--"]
        if not customers_df.empty and "mobile" in customers_df.columns:
            cust_options += customers_df["mobile"].astype(str).tolist()

        selected_customer = st.selectbox("Select Customer (Mobile)", cust_options, key="select_customer_mobile")

        customer_mobile = customer_name = customer_gstin = customer_address = customer_pan = customer_state = ""
        if selected_customer != "--New Customer--" and not customers_df.empty:
            # customers_df may store mobiles as ints; compare as string
            try:
                cust = customers_df[customers_df["mobile"].astype(str) == str(selected_customer)].iloc[0]
                customer_mobile = str(cust.get("mobile", "") or "")
                customer_name = cust.get("name", "") or ""
                # try common gstin column names
                customer_gstin = cust.get("gstin", "") or cust.get("customer_gstin", "") or ""
                customer_address = cust.get("address", "") or ""
                customer_pan = cust.get("pan", "") or ""
                customer_state = cust.get("state", "") if "state" in cust.index else ""
            except Exception:
                customer_mobile = str(selected_customer)

        # --- Indian States & UTs list for dropdown (common)
        INDIAN_STATES_UTS = [
            "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana","Himachal Pradesh",
            "Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland",
            "Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal",
            "Andaman and Nicobar Islands","Chandigarh","Dadra and Nagar Haveli and Daman and Diu","Delhi","Jammu and Kashmir",
            "Ladakh","Lakshadweep","Puducherry"
        ]

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            customer_mobile = st.text_input("Mobile", value=customer_mobile, key="customer_mobile")
            customer_gstin = st.text_input("GSTIN", value=customer_gstin, key="customer_gstin")
        with c2:
            customer_name = st.text_input("Name", value=customer_name, key="customer_name")
            customer_pan = st.text_input("PAN", value=customer_pan, key="customer_pan")
        with c3:
            customer_address = st.text_area("Address", value=customer_address, height=120, key="customer_address")

        # State dropdown (mandatory)
        st.markdown("**State (Customer)** — *select the customer's state / UT (mandatory)*")
        default_state = (customer_state or "").strip()
        # compute index safely
        try:
            idx = INDIAN_STATES_UTS.index(default_state) + 1 if default_state in INDIAN_STATES_UTS else 0
        except Exception:
            idx = 0
        state = st.selectbox("State", options=["--Select State--"] + INDIAN_STATES_UTS, index=idx, key="customer_state")

        update_customer_master_checkbox = st.checkbox("Update Customer Master", value=False, key="update_customer_master")

    # ---------- Compact Add Item section (labels visible, reduced widths) ----------
    with st.expander("💍 Add Invoice Items", expanded=True):
        st.markdown("##### ➕ Add Item")
        c1, c2, c3, c4, c5, c6, c7 = st.columns([0.7, 0.4, 1.0, 0.7, 0.4, 0.5, 0.5])

        categories = ["Gold Ornaments", "Silver Ornaments", "Diamond Ornaments", "Polish Charge"]
        with c1:
            cat = st.selectbox("Category", categories, key="new_item_cat")

        # defaults
        purity = ""
        stock_id = None
        unit = ""
        avail = None
        matching = pd.DataFrame()
        description = ""
        making = 0.0
        qty = 0.0
        rate = 0.0

        if cat == "Polish Charge":
            with c2:
                qty = st.number_input("Qty (Grms)", 0.0, step=0.01, key="new_item_qty_polish")
            with c3:
                rate = st.number_input("Rate (₹)", 0.0, step=0.01, key="new_item_rate_polish")
            with c5:
                st.markdown("Unit: Grms")
            unit = "Grms"
            purity = ""
            stock_id = None
            avail = None

        else:
            with c2:
                unit = "Grms" if cat in ["Gold Ornaments", "Silver Ornaments"] else "Ct"
                purities = stocks_df[stocks_df["category"] == cat]["purity"].dropna().unique().tolist()
                purity = st.selectbox("Purity", purities if purities else ["Standard"], key="new_item_purity")
            with c3:
                matching = stocks_df[(stocks_df["category"] == cat) & (stocks_df["purity"] == purity)]
                stock_opts = ["--Select Stock--"]
                if not matching.empty:
                    stock_opts += matching.apply(lambda r: f"{r['id']} | {r.get('description','')} | Avail:{r.get('quantity',0)}", axis=1).tolist()
                stock_sel = st.selectbox("Stock", stock_opts, key="new_item_stocksel")
                if stock_sel != "--Select Stock--":
                    try:
                        stock_id = int(str(stock_sel).split("|")[0].strip())
                    except Exception:
                        stock_id = None
            with c4:
                description = st.text_input("Description", key="create_item_description")
            with c5:
                qty = st.number_input(f"Qty ({unit})", 0.0, step=0.01, key="new_item_qty")
            with c6:
                rate = st.number_input("Rate (₹)", 0.0, step=0.01, key="new_item_rate")
            with c7:
                making = st.number_input("Making (₹)", 0.0, step=0.01, key="new_item_making")

            avail = None
            if stock_id is not None and not matching.empty:
                row = matching[matching["id"] == stock_id]
                if not row.empty:
                    try:
                        avail = float(row.iloc[0].get("quantity", 0.0) or 0.0)
                    except Exception:
                        avail = None
            if avail is not None and qty > avail:
                st.warning(f"⚠️ Qty exceeds available stock ({avail})")

        # Add item button (compact)
        if st.button("Add Item", use_container_width=True, key="btn_add_item"):
            if cat != "Polish Charge" and not (description and str(description).strip()):
                st.warning("Enter description for the selected stock item.")
            elif safe_float(qty, 0.0) <= 0:
                st.warning("Enter valid quantity (> 0).")
            elif (cat != "Polish Charge") and (avail is not None) and safe_float(qty, 0.0) > avail:
                st.warning(f"Cannot add: qty {qty} > stock {avail}.")
            else:
                if cat == "Polish Charge":
                    line_item_name = "Polish Charge"
                    line_making = 0.0
                    unit = "Grms"
                else:
                    line_item_name = str(description).strip()
                    line_making = safe_float(making, 0.0)

                line_amount = safe_float(qty) * safe_float(rate) + safe_float(line_making)
                st.session_state.setdefault("invoice_items", [])
                st.session_state["invoice_items"].append(
                    {
                        "stock_id": int(stock_id) if stock_id is not None else None,
                        "category": cat,
                        "purity": purity,
                        "item_name": line_item_name,
                        "qty": float(qty),
                        "unit": unit or "",
                        "rate": float(rate),
                        "making": float(line_making),
                        "amount": float(line_amount),
                    }
                )
                st.success("✅ Item added")

    # ---------- Editable Invoice Items table ----------
    st.markdown("---")
    st.markdown("##### 📋 Invoice Items")
    items = st.session_state.get("invoice_items", [])
    if not items:
        st.info("No items added yet.")
    else:
        df_items = pd.DataFrame(items).reset_index(drop=True)
        edited_df = None
        try:
            edited_df = st.data_editor(df_items, num_rows="dynamic", use_container_width=True, key="editor_items")
        except Exception:
            st.dataframe(df_items, use_container_width=True, height=220)
            edited_df = None

        if edited_df is not None:
            if st.button("Apply table edits", use_container_width=True, key="btn_apply_table_edits"):
                new_records = []
                for _, r in edited_df.iterrows():
                    rec = {
                        "stock_id": int(r.get("stock_id")) if pd.notna(r.get("stock_id")) and r.get("stock_id") != "" else None,
                        "category": r.get("category") or "",
                        "purity": r.get("purity") or "",
                        "item_name": r.get("item_name") or "",
                        "qty": safe_float(r.get("qty"), 0.0),
                        "unit": r.get("unit") or "",
                        "rate": safe_float(r.get("rate"), 0.0),
                        "making": safe_float(r.get("making"), 0.0),
                        "amount": safe_float(r.get("amount"), 0.0),
                    }
                    new_records.append(rec)
                st.session_state["invoice_items"] = new_records
                st.success("Applied edits to invoice items.")

        remove_index = st.number_input(
            "Remove item index (1-based)", min_value=0, max_value=len(items), value=0, step=1, key="remove_index"
        )
        if st.button("Remove Item", use_container_width=True, key="btn_remove_item") and remove_index > 0:
            idx = remove_index - 1
            if 0 <= idx < len(st.session_state.get("invoice_items", [])):
                st.session_state["invoice_items"].pop(idx)
                st.success("Item removed.")

    # ---------- Totals, GST, TCS & Payment ----------
    with st.expander("💰 GST, TCS & Payment Details", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            # default value 3.0 kept as you used
            gst_rate_widget = st.number_input("GST (%)", 3.0, step=0.1, key="gst_rate")
        with c2:
            gst_type_widget = st.selectbox("GST Type", ["Intra-State", "Inter-State"], key="gst_type")
        with c3:
            tcs_rate_widget = st.number_input("TCS (%)", 0.0, step=0.1, key="tcs_rate")

        subtotal = sum([safe_float(x.get("amount", 0)) for x in st.session_state.get("invoice_items", [])])
        # prefer session values if present (keeps UI inputs consistent)
        gst_rate = safe_float(st.session_state.get("gst_rate", gst_rate_widget))
        tcs_rate = safe_float(st.session_state.get("tcs_rate", tcs_rate_widget))

        gst_total = subtotal * (gst_rate / 100.0)
        tcs_total = subtotal * (tcs_rate / 100.0)
        grand_total = subtotal + gst_total + tcs_total

        st.markdown(f"**Subtotal:** ₹{subtotal:,.2f}    **GST:** ₹{gst_total:,.2f}    **TCS:** ₹{tcs_total:,.2f}    **Grand:** ₹{grand_total:,.2f}")

        pay_col1, pay_col2, pay_col3 = st.columns([1, 1, 1])
        with pay_col1:
            pay_received = st.number_input("Payment Received (₹)", 0.0, step=0.01, key="pay_received")
        with pay_col2:
            pay_mode = st.selectbox("Mode", ["Cash", "Card", "UPI", "Bank Transfer", "Cheque"], key="pay_mode")
        with pay_col3:
            pay_note = st.text_input("Payment Note", key="pay_note")

    # ---------- Save Invoice (REPLACEMENT: ensures payments.customer_name is set + writes sale_register and customer_state) ----------
    if st.button("💾 Save & Generate Invoice", use_container_width=True, key="btn_save_invoice"):
        try:
            import json as _json  # local alias

            # Defensive reads from session_state / form fields
            items = st.session_state.get("invoice_items", []) or []
            subtotal = sum([safe_float(x.get("amount", 0)) for x in items])
            gst_rate = safe_float(st.session_state.get("gst_rate", 0.0))
            tcs_rate = safe_float(st.session_state.get("tcs_rate", 0.0))
            pay_mode = st.session_state.get("pay_mode", "") or ""
            pay_received = safe_float(st.session_state.get("pay_received", 0.0))
            pay_note = st.session_state.get("pay_note", "")

            # read customer fields (from session keys / widgets)
            customer_mobile_val = (st.session_state.get("customer_mobile", "") or "").strip()
            customer_name_val = (st.session_state.get("customer_name", "") or "").strip()
            customer_gstin_val = (st.session_state.get("customer_gstin", "") or "").strip()
            customer_address_val = (st.session_state.get("customer_address", "") or "").strip()
            customer_pan_val = (st.session_state.get("customer_pan", "") or "").strip()
            customer_state_val = (st.session_state.get("customer_state", "") or "").strip()

            # Validate state (mandatory)
            if not customer_state_val or customer_state_val == "--Select State--":
                st.error("Customer state is mandatory. Please select the customer's State / UT before saving the invoice.")
                st.stop()

            # Compute GST & TCS
            gst_total = subtotal * (gst_rate / 100.0)
            # apply TCS rule for cash > 500k
            tcs_applied = False
            if str(pay_mode or "").strip().lower() == "cash" and subtotal > (500000 - 1e-9):
                if tcs_rate < 1.0:
                    st.info("Cash sale > ₹500,000 detected — applying 1% TCS automatically for this invoice.")
                tcs_rate = max(1.0, tcs_rate)
                tcs_applied = True
            tcs_total = subtotal * (tcs_rate / 100.0)
            grand_total = subtotal + gst_total + tcs_total

            if tcs_applied:
                st.info(f"Applied TCS: ₹{tcs_total:,.2f} (@{tcs_rate:.2f}%) — Grand Total updated: ₹{grand_total:,.2f}")
            else:
                st.info(f"TCS: ₹{tcs_total:,.2f} (@{tcs_rate:.2f}%) — Grand Total: ₹{grand_total:,.2f}")

            # invoice date safe
            try:
                inv_dt = inv_date if isinstance(inv_date, datetime.date) else (pd.to_datetime(inv_date).date() if inv_date else datetime.date.today())
            except Exception:
                inv_dt = datetime.date.today()

            if inv_dt.month >= 4:
                fy_start = datetime.date(inv_dt.year, 4, 1)
                fy_end = datetime.date(inv_dt.year + 1, 3, 31)
            else:
                fy_start = datetime.date(inv_dt.year - 1, 4, 1)
                fy_end = datetime.date(inv_dt.year, 3, 31)

            # Check cash collections in FY (to enforce PAN requirement)
            cash_collections_fy = 0.0
            try:
                conn_for_check = get_conn()
                cur = conn_for_check.cursor()
                cur.execute(
                    """
                    SELECT COALESCE(SUM(amount),0) FROM payments
                    WHERE customer_mobile = ? AND lower(COALESCE(mode,'')) = 'cash' AND date BETWEEN ? AND ?
                    """,
                    (customer_mobile_val, str(fy_start), str(fy_end)),
                )
                rsum = cur.fetchone()
                cash_collections_fy = safe_float(rsum[0], 0.0) if rsum else 0.0
                # include current payment if cash
                if str(pay_mode or "").strip().lower() == "cash" and pay_received > 0:
                    cash_collections_fy += pay_received
            except Exception:
                cash_collections_fy = cash_collections_fy or 0.0
            finally:
                try:
                    conn_for_check.close()
                except Exception:
                    pass

            if cash_collections_fy > (200000 - 1e-9) and (not customer_pan_val or str(customer_pan_val).strip() == ""):
                st.error("PAN is mandatory: customer's cash collections in this financial year exceed ₹200,000. Please enter PAN before saving the invoice.")
                st.stop()

            # --- Optionally update customer master (ensure 'state' column exists and save) ---
            if st.session_state.get("update_customer_master", False):
                try:
                    conn_up = get_conn()
                    cur_up = conn_up.cursor()
                    cur_up.execute("PRAGMA table_info(customers)")
                    existing_cols = [r[1] for r in cur_up.fetchall()]

                    # Add columns if missing (safe)
                    if "pan" not in existing_cols:
                        try:
                            cur_up.execute("ALTER TABLE customers ADD COLUMN pan TEXT")
                        except Exception:
                            pass
                    if "email" not in existing_cols:
                        try:
                            cur_up.execute("ALTER TABLE customers ADD COLUMN email TEXT")
                        except Exception:
                            pass
                    if "gstin" not in existing_cols:
                        try:
                            cur_up.execute("ALTER TABLE customers ADD COLUMN gstin TEXT")
                        except Exception:
                            pass
                    if "address" not in existing_cols:
                        try:
                            cur_up.execute("ALTER TABLE customers ADD COLUMN address TEXT")
                        except Exception:
                            pass
                    if "state" not in existing_cols:
                        try:
                            cur_up.execute("ALTER TABLE customers ADD COLUMN state TEXT")
                        except Exception:
                            pass
                    conn_up.commit()
                except Exception as e:
                    st.warning(f"Customer table schema check failed: {e}")
                finally:
                    try:
                        conn_up.close()
                    except Exception:
                        pass

                # Save / update customer row (use save_customer then ensure state & pan updated)
                if customer_mobile_val:
                    try:
                        # save_customer may create or update a row; we call it if available
                        if "save_customer" in globals():
                            try:
                                save_customer(customer_mobile_val, customer_name_val or "", customer_gstin_val or "", customer_address_val or "")
                            except Exception:
                                pass
                        conn_up2 = get_conn()
                        cur2 = conn_up2.cursor()
                        if customer_pan_val:
                            try:
                                cur2.execute("UPDATE customers SET pan=? WHERE mobile=?", (str(customer_pan_val).upper(), customer_mobile_val))
                            except Exception:
                                pass
                        if customer_state_val and customer_state_val != "--Select State--":
                            try:
                                cur2.execute("UPDATE customers SET state=? WHERE mobile=?", (customer_state_val, customer_mobile_val))
                            except Exception:
                                pass
                        conn_up2.commit()
                    except Exception as e2:
                        st.warning(f"Could not update customer master: {e2}")
                    finally:
                        try:
                            conn_up2.close()
                        except Exception:
                            pass

            # Build kwargs and pass totals (gst_amount, tcs_amount, grand_total) to insert function
            full_kwargs = dict(
                invoice_no_requested=invoice_no,
                date_str=str(inv_dt),
                customer_mobile=customer_mobile_val or None,
                customer_name=customer_name_val or None,
                customer_gstin=customer_gstin_val or None,
                customer_address=customer_address_val or None,
                customer_pan=customer_pan_val or None,
                customer_state=customer_state_val or None,
                gst_rate=gst_rate,
                gst_amount=round(gst_total, 2),
                gst_type=st.session_state.get("gst_type", gst_type_widget),
                tcs_rate=tcs_rate,
                tcs_amount=round(tcs_total, 2),
                items=items,
                subtotal_amount=round(subtotal, 2),
                grand_total=round(grand_total, 2),
                payment_status="Paid" if safe_float(pay_received, 0.0) > 0 else "Unpaid",
                payment_mode=pay_mode or None,
                payment_received=safe_float(pay_received, 0.0),
                payment_date=str(datetime.date.today()),
                payment_note=pay_note,
            )

            # ---------- SAFELY CALL insert_invoice_and_items ----------
            inv_no = None
            if "insert_invoice_and_items" in globals():
                try:
                    inv_no = insert_invoice_and_items(**full_kwargs)
                except TypeError as te:
                    st.warning(f"insert_invoice_and_items rejected some kwargs: {te}. Retrying with trimmed args...")
                    try:
                        import inspect
                        sig = inspect.signature(insert_invoice_and_items)
                        allowed = set(sig.parameters.keys())
                        filtered = {k: v for k, v in full_kwargs.items() if k in allowed}
                        if not filtered:
                            raise RuntimeError("No common parameters found between provided kwargs and function signature.")
                        inv_no = insert_invoice_and_items(**filtered)
                        st.info("Saved invoice using filtered arguments compatible with insert_invoice_and_items().")
                    except Exception as e2:
                        st.warning(f"Signature inspection failed or not compatible: {e2}. Using minimal fallback args.")
                        try:
                            minimal = dict(
                                invoice_no_requested=full_kwargs.get("invoice_no_requested"),
                                date_str=full_kwargs.get("date_str"),
                                customer_mobile=full_kwargs.get("customer_mobile"),
                                items=full_kwargs.get("items"),
                                payment_status=full_kwargs.get("payment_status"),
                                payment_mode=full_kwargs.get("payment_mode"),
                                payment_received=full_kwargs.get("payment_received"),
                            )
                            inv_no = insert_invoice_and_items(**minimal)
                        except Exception as e3:
                            st.error(f"insert_invoice_and_items ultimately failed: {e3}")
                            raise
            else:
                st.warning("insert_invoice_and_items() not available in this environment — invoice will not be written to primary invoice table.")
                # create a fallback invoice id for sale_register usage
                inv_no = invoice_no

            # ------------------ WRITE/UPDATE sale_register (all invoice fields, include customer_state) ------------------
            try:
                conn_sr = get_conn()
                cur_sr = conn_sr.cursor()
                # ensure sale_register has customer_state column and create table if missing
                cur_sr.execute("PRAGMA table_info(sale_register)")
                sr_cols = [r[1] for r in cur_sr.fetchall()]
                if not sr_cols:
                    # create table if entirely missing
                    cur_sr.execute("""
                        CREATE TABLE IF NOT EXISTS sale_register (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            invoice_no TEXT UNIQUE,
                            invoice_date TEXT,
                            customer_name TEXT,
                            customer_mobile TEXT,
                            customer_gstin TEXT,
                            customer_pan TEXT,
                            customer_address TEXT,
                            customer_state TEXT,
                            subtotal REAL,
                            gst_rate REAL,
                            gst_amount REAL,
                            tcs_rate REAL,
                            tcs_amount REAL,
                            grand_total REAL,
                            payment_status TEXT,
                            payment_mode TEXT,
                            payment_received REAL,
                            payment_date TEXT,
                            payment_note TEXT,
                            items_json TEXT,
                            created_at TEXT,
                            updated_at TEXT
                        )
                    """)
                    conn_sr.commit()
                    # refresh sr_cols
                    cur_sr.execute("PRAGMA table_info(sale_register)")
                    sr_cols = [r[1] for r in cur_sr.fetchall()]
                else:
                    # add customer_state if missing
                    if "customer_state" not in sr_cols:
                        try:
                            cur_sr.execute("ALTER TABLE sale_register ADD COLUMN customer_state TEXT")
                            conn_sr.commit()
                        except Exception:
                            pass

                # prepare values
                sr_invoice_no = inv_no
                sr_invoice_date = str(inv_dt)
                sr_customer_name = full_kwargs.get("customer_name") or None
                sr_customer_mobile = full_kwargs.get("customer_mobile") or None
                sr_customer_gstin = full_kwargs.get("customer_gstin") or None
                sr_customer_pan = full_kwargs.get("customer_pan") or None
                sr_customer_address = full_kwargs.get("customer_address") or None
                sr_customer_state = full_kwargs.get("customer_state") or None
                sr_subtotal = float(full_kwargs.get("subtotal_amount", 0.0) or 0.0)
                sr_gst_rate = float(full_kwargs.get("gst_rate", 0.0) or 0.0)
                sr_gst_amount = float(full_kwargs.get("gst_amount", 0.0) or 0.0)
                sr_tcs_rate = float(full_kwargs.get("tcs_rate", 0.0) or 0.0)
                sr_tcs_amount = float(full_kwargs.get("tcs_amount", 0.0) or 0.0)
                sr_grand_total = float(full_kwargs.get("grand_total", 0.0) or 0.0)
                sr_payment_status = full_kwargs.get("payment_status")
                sr_payment_mode = full_kwargs.get("payment_mode") or None
                sr_payment_received = float(full_kwargs.get("payment_received", 0.0) or 0.0)
                sr_payment_date = full_kwargs.get("payment_date") or None
                sr_payment_note = full_kwargs.get("payment_note") or None
                sr_items_json = _json.dumps(items, default=str)
                now_ts = str(datetime.datetime.now())

                # upsert
                cur_sr.execute("SELECT id FROM sale_register WHERE invoice_no = ? LIMIT 1", (sr_invoice_no,))
                existing_sr = cur_sr.fetchone()
                if existing_sr:
                    cur_sr.execute("""
                        UPDATE sale_register SET
                            invoice_date=?, customer_name=?, customer_mobile=?, customer_gstin=?, customer_pan=?, customer_address=?, customer_state=?,
                            subtotal=?, gst_rate=?, gst_amount=?, tcs_rate=?, tcs_amount=?, grand_total=?,
                            payment_status=?, payment_mode=?, payment_received=?, payment_date=?, payment_note=?,
                            items_json=?, updated_at=?
                        WHERE invoice_no=?
                    """, (sr_invoice_date, sr_customer_name, sr_customer_mobile, sr_customer_gstin, sr_customer_pan, sr_customer_address, sr_customer_state,
                          sr_subtotal, sr_gst_rate, sr_gst_amount, sr_tcs_rate, sr_tcs_amount, sr_grand_total,
                          sr_payment_status, sr_payment_mode, sr_payment_received, sr_payment_date, sr_payment_note,
                          sr_items_json, now_ts, sr_invoice_no))
                else:
                    cur_sr.execute("""
                        INSERT INTO sale_register (
                            invoice_no, invoice_date, customer_name, customer_mobile, customer_gstin, customer_pan, customer_address, customer_state,
                            subtotal, gst_rate, gst_amount, tcs_rate, tcs_amount, grand_total,
                            payment_status, payment_mode, payment_received, payment_date, payment_note,
                            items_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (sr_invoice_no, sr_invoice_date, sr_customer_name, sr_customer_mobile, sr_customer_gstin, sr_customer_pan, sr_customer_address, sr_customer_state,
                          sr_subtotal, sr_gst_rate, sr_gst_amount, sr_tcs_rate, sr_tcs_amount, sr_grand_total,
                          sr_payment_status, sr_payment_mode, sr_payment_received, sr_payment_date, sr_payment_note,
                          sr_items_json, now_ts, now_ts))
                conn_sr.commit()
            except Exception as se:
                st.warning(f"Could not write sale_register: {se}")
            finally:
                try:
                    conn_sr.close()
                except Exception:
                    pass

            # ---------- Ensure payments table has customer_name column and insert/update payment row ----------
            try:
                connp = get_conn()
                curp = connp.cursor()
                # safe migration: add column if missing
                try:
                    curp.execute("PRAGMA table_info(payments)")
                    pcols = [r[1] for r in curp.fetchall()]
                    if "customer_name" not in pcols:
                        try:
                            curp.execute("ALTER TABLE payments ADD COLUMN customer_name TEXT")
                            connp.commit()
                        except Exception:
                            pass
                except Exception:
                    pass

                invoice_identifier = inv_no
                pm_amount = float(pay_received) if pay_received else 0.0
                pm_mode = pay_mode or None
                pm_note = pay_note or None
                pm_mobile = full_kwargs.get("customer_mobile") or None
                pm_name = full_kwargs.get("customer_name") or None
                pm_date = str(datetime.date.today())

                if pm_amount and pm_amount > 0:
                    # Check if a payment exists for this invoice (prevent duplicates)
                    curp.execute("SELECT id, amount FROM payments WHERE invoice_no = ? LIMIT 1", (invoice_identifier,))
                    existing = curp.fetchone()
                    if existing:
                        pay_id = existing[0]
                        # Update to ensure customer_name and mobile are set
                        curp.execute("""
                            UPDATE payments
                            SET customer_name = COALESCE(?, customer_name),
                                customer_mobile = COALESCE(?, customer_mobile),
                                amount = ?,
                                mode = ?,
                                note = ?,
                                date = ?
                            WHERE id = ?
                        """, (pm_name, pm_mobile, pm_amount, pm_mode, pm_note, pm_date, pay_id))
                    else:
                        curp.execute("""
                            INSERT INTO payments (date, invoice_no, customer_name, customer_mobile, amount, mode, note, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (pm_date, invoice_identifier, pm_name, pm_mobile, pm_amount, pm_mode, pm_note, str(datetime.datetime.now())))
                    connp.commit()
            except Exception as e:
                st.warning(f"Payment insertion/update failed (customer name may not be saved in payments): {e}")
            finally:
                try:
                    connp.close()
                except Exception:
                    pass

            # Continue: generate PDF, show download, success
            comp = None
            try:
                if "fetch_company" in globals():
                    comp = fetch_company()
            except Exception:
                comp = None

            inv_row = None
            items_rows = None
            try:
                if "read_invoice_from_db" in globals():
                    inv_row, items_rows = read_invoice_from_db(inv_no)
                else:
                    # fallback: try to load from sale_register and/or invoices table
                    try:
                        conn_try = get_conn()
                        df_try = pd.read_sql_query("SELECT * FROM sale_register WHERE invoice_no = ? LIMIT 1", conn_try, params=(inv_no,))
                        if not df_try.empty:
                            inv_row = df_try.iloc[0].to_dict()
                        # items we already have in `items`
                        items_rows = items
                    except Exception:
                        inv_row, items_rows = None, items
                    finally:
                        try:
                            conn_try.close()
                        except Exception:
                            pass
            except Exception:
                inv_row, items_rows = None, items

            pdf_bytes = None
            try:
                if "generate_invoice_pdf" in globals() and inv_row is not None:
                    try:
                        pdf_bytes = generate_invoice_pdf(inv_no, inv_row, items_rows, company_row=comp)
                    except Exception as e:
                        st.warning(f"generate_invoice_pdf() failed: {e}")
                        pdf_bytes = None
            except Exception:
                pdf_bytes = None

            if pdf_bytes:
                try:
                    auto_download_pdf(pdf_bytes, f"{inv_no}.pdf")
                    data_bytes = pdf_bytes.getvalue() if hasattr(pdf_bytes, "getvalue") else bytes(pdf_bytes)
                    st.download_button("Download Invoice PDF", data=data_bytes, file_name=f"{inv_no}.pdf", mime="application/pdf")
                except Exception as e:
                    st.warning(f"Could not prepare PDF download: {e}")

            st.success(f"✅ Invoice {inv_no} saved successfully!")

            # ------------------------
            # AUTO REFRESH / RESET PAGE (unchanged)
            keys_to_clear = [
                "draft_invoice_no", "invoice_items", "inv_date_create",
                "customer_mobile", "customer_name", "customer_gstin", "customer_address", "customer_pan", "customer_state",
                "gst_rate", "gst_type", "tcs_rate", "pay_received", "pay_mode", "pay_note",
                "new_item_cat", "new_item_qty", "new_item_rate", "new_item_making", "new_item_qty_polish",
                "new_item_rate_polish", "new_item_making_polish", "create_item_description", "create_item_description_polish",
                "editor_items", "select_customer_mobile"
            ]
            for k in keys_to_clear:
                st.session_state.pop(k, None)

            # Version-safe rerun to refresh UI
            try:
                st.rerun()
            except AttributeError:
                try:
                    st.experimental_rerun()
                except Exception:
                    st.info("Page will refresh on next action.")

        except Exception as e:
            st.error(f"Error saving invoice: {e}")
            st.error(traceback.format_exc())





# ---------------- Sales Register (Unified Tabs: Sales, Advances, Outstanding, Invoice History, Customer Master, TCS, SFT, GST, Cash, Payments Ledger) ----------------
elif page in ("Report & Register", "🛒 Report & Register"):
    import pandas as pd
    import datetime
    import json
    import streamlit as st
    import traceback
    import re

    # ---------- Header & Theme ----------
    st.markdown("""
        <div style='background:#0A1F44;border-radius:10px;padding:10px 18px;margin-bottom:12px;
                    border:1px solid #0A1F44;box-shadow:0 2px 6px rgba(0,0,0,0.15);'>
            <h3 style='margin:0;color:#ffffff;font-weight:600;'>🛒 Reports & Registers </h3>
            <div style='color:rgba(255,255,255,0.85);font-size:13px;'>Invoices, advances, outstanding, customer master, TCS, SFT, GST and cash register </div>
        </div>
    """, unsafe_allow_html=True)

    # Apply the Outstanding Summary CSS across the page (uniform look & feel)
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] { background-color: #1C1C1C; color: #FFFFFF; font-family: "Inter","Segoe UI",sans-serif; }
        .crm-card { background:#2a2a2a; border-radius:12px; padding:18px 22px; margin-top:14px; margin-bottom:20px; border:1px solid #C0C0C0; box-shadow:0 2px 8px rgba(0,0,0,0.05); }
        .summary-pill { background: linear-gradient(90deg, #0A1F44 0%, #0A1F44 100%); padding:10px 14px; border-radius:8px; color:#C0C0C0; font-weight:600; display:inline-block; width:100%; border-left:4px solid #C0C0C0; margin:10px 0; }
        input, select, textarea { border-radius:8px !important; border:1.5px solid #cbd5e1 !important; padding:0.4rem 0.6rem !important; }
        input:focus, select:focus { border-color:#C0C0C0 !important; box-shadow:0 0 0 2px rgba(192,192,192,0.1); outline:none; }
        div[data-testid="stButton"] > button { background-color:#0A1F44; color:white; border:none; border-radius:8px; padding:6px 16px; font-weight:500; transition:all 0.2s; }
        div[data-testid="stButton"] > button:hover { background-color:#C0C0C0; color:black; }
        [data-testid="stDataFrame"] table { border-collapse:collapse !important; font-size:14px; border-radius:8px !important; }
        [data-testid="stDataFrame"] th { background-color:#0A1F44 !important; color:#C0C0C0 !important; font-weight:600; border-bottom:1px solid #C0C0C0 !important; text-align:left !important; position:sticky !important; top:0; z-index:2; }
        [data-testid="stDataFrame"] td { border-bottom:1px solid #C0C0C0 !important; background-color:#2a2a2a !important; }
        [data-testid="stDataFrame"] tr:nth-child(even) td { background-color:#333333 !important; }
        [data-testid="stDataFrame"] tr:hover td { background-color:#0A1F44 !important; color:#ffffff; transition:0.15s; }
        [data-testid="stMetricValue"] { color:#C0C0C0 !important; font-weight:600 !important; }
        ::-webkit-scrollbar { width:8px; }
        ::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:10px; }
        ::-webkit-scrollbar-thumb:hover { background:#94a3b8; }
        .tooltip { position:relative; display:inline-block; cursor:pointer; }
        .tooltip .tooltiptext { visibility:hidden; width:220px; background-color:#0A1F44; color:#fff; text-align:left; border-radius:8px; padding:6px 10px; position:absolute; z-index:100; bottom:125%; left:50%; margin-left:-110px; opacity:0; transition:opacity 0.3s; font-size:12px; box-shadow:0 4px 10px rgba(0,0,0,0.15); }
        .tooltip:hover .tooltiptext { visibility:visible; opacity:1; }
        /* small helper classes reused by invoice history compact list */
        .status-pill { display:inline-block; min-width:72px; padding:5px 8px; border-radius:999px; font-weight:600; text-align:center; color:#fff; font-size:12px; }
        .status-paid { background:#16a34a; } .status-active { background:#0A84FF; } .status-partial { background:#f59e0b; } .status-cancel { background:#ef4444; } .status-default { background:#6b7280; }
        </style>
    """, unsafe_allow_html=True)

    # ---------- Safe DB migrations (customers.state, sale_register.customer_state) ----------
    try:
        conn_m = get_conn()
        cur_m = conn_m.cursor()
        # ensure customers.state exists
        try:
            cur_m.execute("PRAGMA table_info(customers)")
            cust_cols = [r[1] for r in cur_m.fetchall()]
            if "state" not in cust_cols:
                cur_m.execute("ALTER TABLE customers ADD COLUMN state TEXT")
                conn_m.commit()
        except Exception:
            pass
        # ensure sale_register exists and has customer_state
        try:
            cur_m.execute("PRAGMA table_info(sale_register)")
            sr_cols = [r[1] for r in cur_m.fetchall()]
            if not sr_cols:
                cur_m.execute("""
                    CREATE TABLE IF NOT EXISTS sale_register (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        invoice_no TEXT UNIQUE,
                        invoice_date TEXT,
                        customer_name TEXT,
                        customer_mobile TEXT,
                        customer_gstin TEXT,
                        customer_pan TEXT,
                        customer_address TEXT,
                        customer_state TEXT,
                        subtotal REAL,
                        gst_rate REAL,
                        gst_amount REAL,
                        tcs_rate REAL,
                        tcs_amount REAL,
                        grand_total REAL,
                        payment_status TEXT,
                        payment_mode TEXT,
                        payment_received REAL,
                        payment_date TEXT,
                        payment_note TEXT,
                        items_json TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    )
                """)
                conn_m.commit()
            else:
                if "customer_state" not in sr_cols:
                    cur_m.execute("ALTER TABLE sale_register ADD COLUMN customer_state TEXT")
                    conn_m.commit()
        except Exception:
            pass
    finally:
        try:
            conn_m.close()
        except Exception:
            pass

    # ---------- Load invoice/sale_register snapshot ----------
    try:
        conn = get_conn()
        df_all = pd.read_sql_query("SELECT * FROM sale_register ORDER BY invoice_date DESC, created_at DESC LIMIT 5000", conn)
    except Exception:
        df_all = pd.DataFrame()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # fallback to invoices table if sale_register empty
    if df_all.empty:
        try:
            conn = get_conn()
            df_all = pd.read_sql_query("""
                SELECT invoice_no as invoice_no, date as invoice_date, customer_name, customer_mobile, customer_gstin, customer_pan,
                       subtotal as subtotal, gst_rate as gst_rate, gst_amount as gst_amount, tcs_rate as tcs_rate, tcs_amount as tcs_amount,
                       grand_total as grand_total, payment_status as payment_status, payment_mode as payment_mode, payment_received as payment_received,
                       items_json as items_json
                FROM invoices ORDER BY date DESC LIMIT 5000
            """, conn)
        except Exception:
            df_all = pd.DataFrame()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # normalize date
    if not df_all.empty and "invoice_date" in df_all.columns:
        try:
            df_all["invoice_date"] = pd.to_datetime(df_all["invoice_date"], errors="coerce").dt.date
        except Exception:
            pass

    # extract categories for quick glance if items_json exists
    def extract_categories(items_json):
        try:
            items = json.loads(items_json) if items_json else []
            cats = [it.get("category") for it in items if isinstance(it, dict) and it.get("category")]
            return ", ".join(sorted(set([c for c in cats if c])))
        except Exception:
            return ""
    if not df_all.empty and "items_json" in df_all.columns:
        try:
            df_all["categories"] = df_all["items_json"].apply(extract_categories)
        except Exception:
            df_all["categories"] = ""

    # ---------- Tabs (add Payments Ledger as 10th tab) ----------
    tabs = ["📊 Sales Register","💳 Advances","🧾 Outstanding Summary","📜 Invoice History",
            "👥 Customer Master","💰 TCS Register","🏦 SFT Register","🧾 GST Register",
            "💵 Cash Register","💳 Payments Ledger"]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(tabs)

    # ---------------- Tab 1: Sales Register ----------------
    with tab1:
        st.markdown("<div class='crm-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Sales Register (Invoice & Item-wise)")

        if df_all.empty:
            st.info("No invoices available in sale_register/invoices.")
        else:
            # --- Date range filter (same UX as GST/TCS) ---
            today = datetime.date.today()

            def first_day_of_prev_month(d):
                first_this_month = d.replace(day=1)
                prev_last = first_this_month - datetime.timedelta(days=1)
                return prev_last.replace(day=1)

            def last_day_of_prev_month(d):
                first_this_month = d.replace(day=1)
                prev_last = first_this_month - datetime.timedelta(days=1)
                return prev_last

            range_choice = st.selectbox(
                "Range",
                ["Current month", "Previous month", "This Financial Year", "Custom range", "All"],
                key="sales_reg_range",
            )

            if range_choice == "Current month":
                start_date = today.replace(day=1)
                end_date = today
            elif range_choice == "Previous month":
                start_date = first_day_of_prev_month(today)
                end_date = last_day_of_prev_month(today)
            elif range_choice == "This Financial Year":
                if today.month >= 4:
                    start_date = datetime.date(today.year, 4, 1)
                else:
                    start_date = datetime.date(today.year - 1, 4, 1)
                end_date = today
            elif range_choice == "Custom range":
                c1, c2 = st.columns(2)
                start_date = c1.date_input("Start", value=today.replace(day=1), key="sales_tab_start")
                end_date = c2.date_input("End", value=today, key="sales_tab_end")
            else:
                start_date = None
                end_date = None

            # Work on a copy
            df_sales = df_all.copy()

            # Normalize invoice_date to date if possible
            if "invoice_date" in df_sales.columns:
                try:
                    df_sales["invoice_date"] = pd.to_datetime(df_sales["invoice_date"], errors="coerce").dt.date
                except Exception:
                    pass

            # Apply date filter if selected
            if start_date is not None and end_date is not None:
                try:
                    df_sales = df_sales[(df_sales["invoice_date"] >= start_date) & (df_sales["invoice_date"] <= end_date)]
                except Exception:
                    # If invoice_date missing or filter fails, keep original (defensive)
                    pass

            # Columns to show at invoice-level (only if exist)
            inv_cols = [
                c
                for c in [
                    "invoice_no",
                    "invoice_date",
                    "customer_name",
                    "customer_mobile",
                    "customer_state",
                    "customer_gstin",
                    "customer_pan",
                    "subtotal",
                    "gst_rate",
                    "gst_amount",
                    "tcs_rate",
                    "tcs_amount",
                    "grand_total",
                    "payment_mode",
                    "payment_received",
                    "payment_status",
                    "categories",
                ]
                if c in df_sales.columns
            ]

            # Compute totals defensively
            try:
                total_value = df_sales.get("grand_total", pd.Series(dtype=float)).fillna(0).astype(float).sum()
            except Exception:
                total_value = 0.0

            st.markdown(f"<div class='summary-pill'>Invoices: {len(df_sales)} • Total ₹{total_value:,.2f}</div>", unsafe_allow_html=True)

            with st.expander("📋 Invoice-level Register", expanded=True):
                st.dataframe(df_sales[inv_cols].fillna(""), height=360, use_container_width=True)
                try:
                    csvb = df_sales[inv_cols].to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Invoice CSV", data=csvb, file_name="sales_register_invoices.csv", mime="text/csv", use_container_width=True)
                except Exception:
                    pass

            # Item-wise register: expand items_json into rows for the filtered invoices
            item_rows = []
            for _, inv in df_sales.iterrows():
                meta = {
                    "invoice_no": inv.get("invoice_no"),
                    "invoice_date": inv.get("invoice_date"),
                    "customer_name": inv.get("customer_name"),
                    "customer_mobile": inv.get("customer_mobile"),
                    "customer_state": inv.get("customer_state") if "customer_state" in inv else None,
                    "customer_gstin": inv.get("customer_gstin"),
                    "customer_pan": inv.get("customer_pan"),
                    "subtotal": inv.get("subtotal"),
                    "gst_rate": inv.get("gst_rate"),
                    "gst_amount": inv.get("gst_amount"),
                    "tcs_rate": inv.get("tcs_rate"),
                    "tcs_amount": inv.get("tcs_amount"),
                    "grand_total": inv.get("grand_total"),
                }
                try:
                    items = json.loads(inv.get("items_json") or "[]")
                except Exception:
                    items = []
                if items:
                    for it in items:
                        row = meta.copy()
                        row.update(
                            {
                                "category": it.get("category") or it.get("cat") or it.get("item_category"),
                                "purity": it.get("purity"),
                                "item_name": it.get("item_name") or it.get("description"),
                                "qty": it.get("qty") or it.get("quantity"),
                                "unit": it.get("unit"),
                                "rate": it.get("rate"),
                                "making": it.get("making"),
                                "amount": it.get("amount"),
                            }
                        )
                        item_rows.append(row)
                else:
                    # If no items_json, optionally include one aggregated row per invoice
                    item_rows.append(
                        {
                            **meta,
                            "category": "Uncategorized",
                            "purity": None,
                            "item_name": "Invoice Total",
                            "qty": 1,
                            "unit": None,
                            "rate": meta.get("subtotal") or 0,
                            "making": 0,
                            "amount": meta.get("subtotal") or 0,
                        }
                    )

            df_items = pd.DataFrame(item_rows)

            with st.expander("📦 Item-wise Register", expanded=False):
                if df_items.empty:
                    st.info("No item-level data available.")
                else:
                    # Ensure invoice_date in df_items is date type (already carried from meta)
                    if "invoice_date" in df_items.columns:
                        try:
                            df_items["invoice_date"] = pd.to_datetime(df_items["invoice_date"], errors="coerce").dt.date
                        except Exception:
                            pass

                    st.dataframe(df_items.fillna(""), height=420, use_container_width=True)
                    try:
                        st.download_button("⬇️ Download Item-wise CSV", data=df_items.to_csv(index=False).encode("utf-8"), file_name="sales_register_items.csv", mime="text/csv", use_container_width=True)
                    except Exception:
                        pass
        st.markdown("</div>", unsafe_allow_html=True)


    # ---------------- Tab 2: Advances (exact copy of Advances page, tab-scoped) ----------------
    with tab2:
        import datetime as _dt

        def safe_float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default

        # --- Header ---
        st.markdown("""
            <div style='display:flex;align-items:center;justify-content:space-between;
                        background:linear-gradient(90deg,#0A1F44 0%,#0A1F44 100%);
                        border-radius:10px;padding:12px 18px;margin-bottom:16px;
                        border:1px solid #0A1F44;box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
                <div>
                    <h3 style='margin:0;color:#ffffff;font-weight:600;'>💳 Customer Advances</h3>
                    <div style='color:rgba(255,255,255,0.85);font-size:13px;margin-top:4px;'>Create, allocate, or delete advances easily</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Page-level styling is already applied globally; re-use helpers
        try:
            customers_df = fetch_customers_df() if "fetch_customers_df" in globals() else pd.DataFrame()
        except Exception:
            customers_df = pd.DataFrame()

        mobiles = customers_df["mobile"].astype(str).tolist() if not customers_df.empty else []
        cust_name_map = {row["mobile"]: row.get("name", "") for _, row in customers_df.iterrows()} if not customers_df.empty else {}

        st.markdown("<div class='crm-card'>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns([2.5, 1, 1])
        search_query = s1.text_input("Search by name or mobile", placeholder="Type name or mobile to filter advances...", key="adv_tab_search")
        show_all = s2.checkbox("Show all advances", value=False, key="adv_tab_show_all")
        refresh = s3.button("🔄 Refresh", key="adv_tab_refresh")
        if refresh:
            try:
                st.experimental_rerun()
            except Exception:
                try:
                    st.rerun()
                except Exception:
                    pass

        # ---------------- Create new advance (compact inline) ----------------
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1.4, 0.9, 0.9, 1, 1])
        cust_opts = ["--New--"] + mobiles
        sel = c1.selectbox("Customer", options=cust_opts, key="adv_tab_sel", label_visibility="collapsed")
        if sel != "--New--" and sel in mobiles:
            default_mob = sel
            default_name = cust_name_map.get(sel, "")
        else:
            default_mob = default_name = ""

        mob = c2.text_input("Mobile", value=default_mob, key="adv_tab_mobile", label_visibility="collapsed", placeholder="Mobile")
        name = c3.text_input("Name", value=default_name, key="adv_tab_name", label_visibility="collapsed", placeholder="Name")
        amt = c4.number_input("Amount (₹)", min_value=0.0, step=0.01, format="%.2f", key="adv_tab_amt", label_visibility="collapsed")
        mode = c5.selectbox("Mode", ["--Select--", "Cash", "Card", "UPI", "Bank Transfer", "Cheque"], index=0, key="adv_tab_mode", label_visibility="collapsed")
        note = c6.text_input("Note", key="adv_tab_note", label_visibility="collapsed", placeholder="Note (optional)")

        if st.button("➕ Create Advance", key="adv_tab_create"):
            if not mob:
                st.error("Customer mobile required")
            elif amt <= 0:
                st.error("Enter positive amount")
            else:
                try:
                    if sel == "--New--" and mob and name and "save_customer" in globals():
                        try:
                            save_customer(mob, name, "", "")
                        except Exception:
                            pass
                    adv_id = None
                    if "create_advance_note" in globals():
                        adv_id = create_advance_note(
                            customer_mobile=mob,
                            amount=amt,
                            date=str(_dt.date.today()),
                            mode=(mode if mode != "--Select--" else None),
                            note=note,
                        )
                    st.success(f"Advance #{adv_id} — ₹{amt:,.2f}" if adv_id else f"Advance recorded — ₹{amt:,.2f}")
                    try:
                        st.experimental_rerun()
                    except Exception:
                        try:
                            st.rerun()
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"Could not create advance: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)

        # ---------------- List advances ----------------
        try:
            advs = fetch_advances(only_with_remaining=False) if "fetch_advances" in globals() else pd.DataFrame()
        except Exception:
            advs = pd.DataFrame()

        if not advs.empty:
            try:
                advs["customer_name"] = advs["customer_mobile"].map(cust_name_map).fillna(advs.get("customer_name",""))
            except Exception:
                pass

            if not show_all:
                if "remaining_amount" in advs.columns:
                    advs = advs[advs["remaining_amount"] > 0]
            if search_query:
                q = search_query.strip().lower()
                advs = advs[
                    advs["customer_mobile"].astype(str).str.contains(q) |
                    advs["customer_name"].str.lower().str.contains(q, na=False)
                ]

        if advs.empty:
            st.info("No advances found.")
        else:
            total_rem = advs.get("remaining_amount", pd.Series(dtype=float)).fillna(0).sum()
            st.markdown(f"<div class='summary-pill'>💰 <b>{len(advs)}</b> active advances • Remaining ₹{total_rem:,.2f}</div>", unsafe_allow_html=True)

            # Header row
            hcol1, hcol2, hcol3, hcol4, hcol5, hcol6 = st.columns([0.6, 1.6, 1, 1, 1, 1.4])
            hcol1.markdown("**ID**")
            hcol2.markdown("**Customer**")
            hcol3.markdown("**Amt**")
            hcol4.markdown("**Rem**")
            hcol5.markdown("**Date**")
            hcol6.markdown("**Actions**")

            if "adv_tab_selected_for_alloc" not in st.session_state:
                st.session_state["adv_tab_selected_for_alloc"] = None

            for _, r in advs.sort_values("date", ascending=False).iterrows():
                rid = int(r["id"])
                mobile = r["customer_mobile"]
                cname = r.get("customer_name", "")
                amt_s = f"{safe_float(r.get('amount', 0)):,.2f}"
                rem_s = f"{safe_float(r.get('remaining_amount', 0)):,.2f}"
                date_s = r.get("date", "")

                col1, col2, col3, col4, col5, col6 = st.columns([0.6, 1.6, 1, 1, 1, 1.4])
                col1.write(rid)
                label = f"{cname} — {mobile}" if cname else mobile
                if col2.button(label, key=f"adv_tab_row_{rid}", use_container_width=True):
                    st.session_state["adv_tab_selected_for_alloc"] = rid
                col3.write(f"₹{amt_s}")
                col4.write(f"₹{rem_s}")
                col5.write(date_s)

                with col6:
                    a1, a2 = st.columns([1, 1])
                    if a1.button("Allocate", key=f"adv_tab_alloc_{rid}"):
                        st.session_state["adv_tab_selected_for_alloc"] = rid
                    if a2.button("Delete", key=f"adv_tab_del_{rid}"):
                        try:
                            allocs = fetch_allocations_for_advance(rid) if "fetch_allocations_for_advance" in globals() else pd.DataFrame()
                            delete_advance(rid, force=(not allocs.empty) if "delete_advance" in globals() else False)
                            st.success(f"Advance #{rid} deleted.")
                            if st.session_state.get("adv_tab_selected_for_alloc") == rid:
                                st.session_state["adv_tab_selected_for_alloc"] = None
                            try:
                                st.experimental_rerun()
                            except Exception:
                                try:
                                    st.rerun()
                                except Exception:
                                    pass
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

            # ---------------- Inline Allocation ----------------
            sel_id = st.session_state.get("adv_tab_selected_for_alloc")
            if sel_id:
                if sel_id not in advs["id"].tolist():
                    st.warning("Advance no longer exists.")
                    st.session_state["adv_tab_selected_for_alloc"] = None
                else:
                    sel_row = advs[advs["id"] == sel_id].iloc[0]
                    st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='adv-card'><strong>Allocate Advance #{sel_id}</strong> → {sel_row['customer_mobile']} &nbsp; • Remaining ₹{safe_float(sel_row.get('remaining_amount',0)):,.2f}</div>",
                        unsafe_allow_html=True,
                    )

                    conn = get_conn()
                    try:
                        inv_df = _get_invoice_dues(conn, sel_row["customer_mobile"]) if "_get_invoice_dues" in globals() else pd.DataFrame(columns=["invoice_no","due"])
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass

                    invs = inv_df[inv_df["due"] > 0].copy() if not inv_df.empty else pd.DataFrame(columns=["invoice_no", "due"])
                    inv_opts = ["--Select invoice--"] + invs["invoice_no"].astype(str).tolist()

                    ic1, ic2, ic3 = st.columns([1.6, 1, 1])
                    inv_sel = ic1.selectbox("Invoice", inv_opts, key=f"adv_tab_inline_inv_{sel_id}")
                    default_amt = float(safe_float(sel_row.get("remaining_amount",0)))
                    alloc_amt = ic2.number_input("Amount", min_value=0.0, step=0.01, value=default_amt if default_amt > 0 else 0.0, key=f"adv_tab_inline_amt_{sel_id}")

                    apply_btn = ic3.button("Apply", key=f"adv_tab_apply_{sel_id}")
                    cancel_btn = ic3.button("Cancel", key=f"adv_tab_cancel_{sel_id}")

                    if apply_btn:
                        if inv_sel == "--Select invoice--":
                            st.error("Select invoice")
                        elif alloc_amt <= 0:
                            st.error("Enter positive amount")
                        else:
                            try:
                                if "allocate_advance_to_invoice" in globals():
                                    allocate_advance_to_invoice(
                                        advance_id=sel_id,
                                        invoice_no=inv_sel,
                                        amount=alloc_amt,
                                        date=str(_dt.date.today()),
                                    )
                                    st.success(f"Allocated ₹{alloc_amt:,.2f} to {inv_sel}")
                                else:
                                    st.info("Allocation helper not available in this environment.")
                                st.session_state["adv_tab_selected_for_alloc"] = None
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    try:
                                        st.rerun()
                                    except Exception:
                                        pass
                            except Exception as e:
                                st.error(f"Allocation failed: {e}")

                    if cancel_btn:
                        st.session_state["adv_tab_selected_for_alloc"] = None

    # ---------------- Tab 3: Outstanding Summary (exact implementation you provided) ----------------
    with tab3:
        # --- Header ---
        st.markdown("""
        <div style='display:flex;align-items:center;justify-content:space-between;
        background:linear-gradient(90deg,#0A1F44 0%,#0A1F44 100%);
        border-radius:12px;padding:14px 22px;margin-bottom:16px;
        border:1px solid #C0C0C0;box-shadow:0 2px 6px rgba(0,0,0,0.03);'>
            <div>
                <h3 style='margin:0;color:#C0C0C0;font-weight:600;'>🧾 Outstanding Receivables</h3>
                <p style='margin:0;color:#ffffff;font-size:13px;'>Customer-wise outstanding summary with real-time search and filters</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Load Data ---
        conn = get_conn()
        try:
            sql_invoices = """
                SELECT customer_mobile, customer_name, COALESCE(SUM(grand_total),0) AS total_invoiced
                FROM invoices
                WHERE COALESCE(status,'Active') != 'Cancelled'
                GROUP BY customer_mobile, customer_name
            """
            df_inv = pd.read_sql_query(sql_invoices, conn)

            sql_paid = """
                SELECT inv.customer_mobile AS customer_mobile, COALESCE(SUM(p.amount),0) AS total_paid_applied
                FROM payments p
                JOIN invoices inv ON p.invoice_no = inv.invoice_no
                WHERE COALESCE(p.is_advance,0) = 0
                GROUP BY inv.customer_mobile
            """
            df_paid = pd.read_sql_query(sql_paid, conn)

            if df_inv.empty:
                df_summary = pd.DataFrame(columns=["customer_mobile","customer_name","total_invoiced","total_paid_applied","outstanding"])
            else:
                df_summary = df_inv.merge(df_paid, on="customer_mobile", how="left")
                df_summary["total_paid_applied"] = df_summary["total_paid_applied"].fillna(0.0)
                df_summary["outstanding"] = df_summary["total_invoiced"] - df_summary["total_paid_applied"]
                df_summary = df_summary[df_summary["outstanding"] > 0]
                df_summary = df_summary.sort_values("outstanding", ascending=False).reset_index(drop=True)
        except Exception as e:
            st.error(f"Could not compute outstanding summary: {e}")
            df_summary = pd.DataFrame()
        finally:
            conn.close()

        if df_summary.empty:
            st.info("No invoice data available.")
        else:
            # --- Filters ---
            with st.expander("🔍 Search & Filter", expanded=True):
                c1, c2 = st.columns([2, 1])
                search_query = c1.text_input("Search by name or mobile", placeholder="Start typing to filter...", key="os_search_live")
                min_outstanding = c2.number_input("Min outstanding (₹)", value=0.0, step=1.0, key="os_min_main")

            # --- Apply filters ---
            df_display = df_summary.copy()
            if search_query:
                q = search_query.strip().lower()
                df_display = df_display[
                    df_display["customer_name"].str.lower().str.contains(q, na=False) |
                    df_display["customer_mobile"].astype(str).str.contains(q)
                ]
            if min_outstanding and min_outstanding > 0:
                df_display = df_display[df_display["outstanding"] >= float(min_outstanding)]

            total_invoiced_all = df_display["total_invoiced"].sum()
            total_paid_all = df_display["total_paid_applied"].sum()
            total_outstanding_all = df_display["outstanding"].sum()

            # --- Summary Pill ---
            st.markdown(f"<div class='summary-pill'>💰 <b>{len(df_display)}</b> customers shown • Total Outstanding: ₹{total_outstanding_all:,.2f}</div>", unsafe_allow_html=True)

            # --- KPI Summary ---
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Invoiced", f"₹{total_invoiced_all:,.2f}")
            k2.metric("Total Paid", f"₹{total_paid_all:,.2f}")
            k3.metric("Outstanding", f"₹{total_outstanding_all:,.2f}")

            # --- Main Summary Table ---
            st.markdown("<div class='crm-card'>", unsafe_allow_html=True)
            st.markdown("### 📋 Customer Outstanding Summary")
            st.caption("Click a customer (Name + Mobile) to view detailed invoices below.")

            if "selected_customer" not in st.session_state:
                st.session_state["selected_customer"] = None

            # Loop through the customers and display each one as a clickable button
            for idx, row in df_display.iterrows():
                c1, c2, c3, c4, c5, c6 = st.columns([0.5, 2.5, 1, 1, 1, 1])
                c1.write(f"{idx+1}")
                display_label = f"{row['customer_name']} ({row['customer_mobile']})"
                tooltip_html = f"""
                <div class='tooltip'>
                    {display_label}
                    <span class='tooltiptext'>
                        💰 Total Invoiced: ₹{row['total_invoiced']:,.2f}<br>
                        ✅ Paid: ₹{row['total_paid_applied']:,.2f}<br>
                        ⚠️ Outstanding: ₹{row['outstanding']:,.2f}
                    </span>
                </div>
                """
                if c2.button(display_label, key=f"cust_{idx}", use_container_width=True):
                    st.session_state["selected_customer"] = row["customer_mobile"]

                c3.write(f"₹{row['total_invoiced']:,.2f}")
                c4.write(f"₹{row['total_paid_applied']:,.2f}")
                c5.write(f"₹{row['outstanding']:,.2f}")
                c6.write(row["customer_mobile"])
            st.markdown("</div>", unsafe_allow_html=True)

            # --- Customer Invoice Details ---
            if st.session_state.get("selected_customer"):
                sel_mobile = st.session_state["selected_customer"]
                cust_row = df_display[df_display["customer_mobile"] == sel_mobile].iloc[0]
                st.markdown(
                    f"<div class='crm-card'><h4 style='color:#C0C0C0;margin-bottom:8px;'>📑 Invoices for {cust_row['customer_name']} ({sel_mobile})</h4>",
                    unsafe_allow_html=True,
                )

                conn = get_conn()
                try:
                    inv_detail = pd.read_sql_query(
                        """
                        SELECT invoice_no, date, grand_total,
                               COALESCE((SELECT SUM(amount) FROM payments WHERE invoice_no=i.invoice_no),0) AS paid,
                               (COALESCE(grand_total,0) - COALESCE((SELECT SUM(amount) FROM payments WHERE invoice_no=i.invoice_no),0)) AS balance
                        FROM invoices i
                        WHERE i.customer_mobile=?
                        ORDER BY date DESC
                        """,
                        conn,
                        params=(sel_mobile,),
                    )
                finally:
                    conn.close()

                if inv_detail.empty:
                    st.info("No invoices found for this customer.")
                else:
                    st.dataframe(
                        inv_detail.style.format({
                            "grand_total": "{:,.2f}",
                            "paid": "{:,.2f}",
                            "balance": "{:,.2f}"
                        }).hide(axis="index"),
                        height=350,
                    )

                if st.button("Close Detail View", key="close_cust_detail"):
                    st.session_state["selected_customer"] = None
                st.markdown("</div>", unsafe_allow_html=True)

            # --- Download Button ---
            st.download_button(
                "📥 Download Outstanding Summary CSV",
                data=df_display.to_csv(index=False),
                file_name="outstanding_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ---------------- Tab 4: Invoice History (replaced with tab-scoped copy of Invoice History page) ----------------
    with tab4:
        st.markdown("### 📜 Invoice History ")

        # --- Load invoices from sale_register (fall back to invoices table) ---
        try:
            conn = get_conn()
            try:
                invoices_df = pd.read_sql_query("""
                    SELECT invoice_no, invoice_date as date, customer_name, customer_mobile, 
                           COALESCE(customer_state,'') AS customer_state, COALESCE(grand_total,0) AS grand_total,
                           COALESCE(payment_status,'') AS status, COALESCE(items_json,'') AS items_json
                    FROM sale_register
                    ORDER BY invoice_date DESC, created_at DESC
                """, conn)
            except Exception:
                invoices_df = pd.read_sql_query("""
                    SELECT invoice_no, date, customer_name, customer_mobile, '' AS customer_state,
                           COALESCE(grand_total,0) AS grand_total, COALESCE(status,'') AS status, '' AS items_json
                    FROM invoices
                    ORDER BY date DESC
                """, conn)
            invoices_df["date"] = pd.to_datetime(invoices_df["date"], errors="coerce").dt.date
        except Exception as e:
            st.error(f"Could not load invoices: {e}")
            invoices_df = pd.DataFrame()
        finally:
            try:
                conn.close()
            except Exception:
                pass

        if invoices_df.empty:
            st.info("No invoices recorded.")
        else:
            search = st.text_input("🔍 Search invoice no / customer / mobile", placeholder="Type invoice number, customer name or mobile...", key="ih_tab_search")

            df_display = invoices_df.copy()
            if search and search.strip():
                q = search.strip().lower()
                df_display = df_display[
                    df_display["customer_name"].astype(str).str.lower().str.contains(q, na=False)
                    | df_display["customer_mobile"].astype(str).str.contains(q)
                    | df_display["invoice_no"].astype(str).str.contains(q)
                ]
                view_label = f"Search Results ({len(df_display)})"
            else:
                df_display = df_display.head(50)
                view_label = f"Latest {len(df_display)} Invoices"

            total_amt = float(df_display.get("grand_total", pd.Series(dtype=float)).fillna(0).sum())
            st.markdown(f"<div class='summary-pill'>📅 {view_label} • Total Value ₹{total_amt:,.2f}</div>", unsafe_allow_html=True)

            st.markdown("<div class='crm-card'>", unsafe_allow_html=True)
            st.markdown("### 🧾 Invoice List")

            if "tab_ih_selected_invoice" not in st.session_state:
                st.session_state["tab_ih_selected_invoice"] = None

            def status_class(s):
                if not s: return "status-default"
                sl = str(s).strip().lower()
                if sl in ("paid","completed","closed"): return "status-paid"
                if sl in ("active","open","unpaid","pending"): return "status-active"
                if sl in ("partial","partial paid","partially paid"): return "status-partial"
                if sl in ("cancelled","canceled","void"): return "status-cancel"
                return "status-default"

            for idx, r in df_display.iterrows():
                c1, c2, c3, c4, c5 = st.columns([1, 2.5, 1, 1, 0.9])
                with c1:
                    if st.button(f"{r['invoice_no']}", key=f"tab_inv_{r['invoice_no']}"):
                        st.session_state["tab_ih_selected_invoice"] = r["invoice_no"]
                        try:
                            st.experimental_rerun()
                        except Exception:
                            try:
                                st.rerun()
                            except Exception:
                                pass
                c2.write(r.get("customer_name", ""))
                c3.write(r.get("date", ""))
                c4.write(f"₹{(r.get('grand_total') or 0):,.2f}")
                pill_cls = status_class(r.get("status", ""))
                c5.markdown(f"<div class='status-pill {pill_cls}'>{r.get('status','')}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state.get("tab_ih_selected_invoice"):
                sel_invoice = st.session_state["tab_ih_selected_invoice"]
                st.markdown(f"<div class='crm-card'><h4 style='color:#4EA8FF;margin-bottom:8px;'>📑 Invoice Details — {sel_invoice}</h4>", unsafe_allow_html=True)

                inv_row, items_rows = None, []
                try:
                    if "read_invoice_from_db" in globals():
                        inv_row, items_rows = read_invoice_from_db(sel_invoice)
                    else:
                        conn2 = get_conn()
                        try:
                            hdr = pd.read_sql_query("SELECT * FROM invoices WHERE invoice_no=? LIMIT 1", conn2, params=(sel_invoice,))
                            if not hdr.empty:
                                inv_row = hdr.iloc[0].to_dict()
                            items_rows = pd.read_sql_query("SELECT * FROM invoice_items WHERE invoice_no=?", conn2, params=(sel_invoice,)).to_dict('records')
                        finally:
                            try:
                                conn2.close()
                            except Exception:
                                pass
                except Exception as e:
                    st.error(f"Error reading invoice from DB: {e}")

                try:
                    conn3 = get_conn(); cur = conn3.cursor()
                    cur.execute("PRAGMA table_info(invoice_items)")
                    cols = [r[1] for r in cur.fetchall()]
                    desc_col = next((c for c in ["description", "item_name", "product", "details"] if c in cols), None)
                    qty_col = "quantity" if "quantity" in cols else "qty" if "qty" in cols else None
                    rate_col = "rate" if "rate" in cols else "price" if "price" in cols else None
                    amt_col = "amount" if "amount" in cols else "total" if "total" in cols else None
                    valid_cols = [c for c in [desc_col, qty_col, rate_col, amt_col] if c]
                    if valid_cols:
                        select_clause = ", ".join(valid_cols)
                        items_df = pd.read_sql_query(f"SELECT {select_clause} FROM invoice_items WHERE invoice_no=?", conn3, params=(sel_invoice,))
                    else:
                        items_df = pd.DataFrame(columns=["Item","Qty","Rate","Amount"])
                    payments_df = pd.read_sql_query("SELECT date, amount, mode, customer_name, customer_mobile FROM payments WHERE invoice_no=?", conn3, params=(sel_invoice,))
                except Exception as e:
                    st.error(f"Error fetching invoice details: {e}")
                    items_df, payments_df = pd.DataFrame(), pd.DataFrame()
                finally:
                    try:
                        conn3.close()
                    except Exception:
                        pass

                if items_df is not None and not items_df.empty:
                    st.markdown("**🧾 Line Items**")
                    fmt = {}
                    if rate_col: fmt[rate_col] = "₹{:,.2f}"
                    if amt_col: fmt[amt_col] = "₹{:,.2f}"
                    st.dataframe(items_df.style.format(fmt))
                else:
                    st.info("No line items found.")

                if payments_df is not None and not payments_df.empty:
                    st.markdown("**💳 Payments**")
                    st.dataframe(payments_df.style.format({"amount":"₹{:,.2f}"}))
                else:
                    st.info("No payments recorded.")

                pdf_bytes = None
                try:
                    comp = fetch_company() if "fetch_company" in globals() else None
                    if "generate_invoice_pdf" in globals() and inv_row is not None:
                        try:
                            pdf_bytes = generate_invoice_pdf(sel_invoice, inv_row, items_rows, company_row=comp)
                        except Exception as e:
                            st.warning(f"generate_invoice_pdf() failed: {e}")
                            pdf_bytes = None
                except Exception:
                    pdf_bytes = None

                if pdf_bytes:
                    try:
                        data_bytes = pdf_bytes.getvalue() if hasattr(pdf_bytes, "getvalue") else bytes(pdf_bytes)
                        st.download_button(label="⬇️ Download Invoice PDF", data=data_bytes, file_name=f"{sel_invoice}.pdf", mime="application/pdf", key=f"tab_dl_inv_pdf_{sel_invoice}", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not prepare PDF download: {e}")
                else:
                    try:
                        if inv_row:
                            header_csv = pd.DataFrame([inv_row]).to_csv(index=False)
                            st.download_button("⬇️ Download Invoice Header (CSV)", data=header_csv.encode("utf-8"), file_name=f"{sel_invoice}_header.csv", mime="text/csv", key=f"tab_dl_inv_hdr_{sel_invoice}")
                        if isinstance(items_rows, list) and items_rows:
                            items_df_fb = pd.DataFrame(items_rows)
                            items_csv = items_df_fb.to_csv(index=False)
                            st.download_button("⬇️ Download Invoice Items (CSV)", data=items_csv.encode("utf-8"), file_name=f"{sel_invoice}_items.csv", mime="text/csv", key=f"tab_dl_inv_items_{sel_invoice}")
                    except Exception:
                        pass

                if st.button("❌ Close Details", key=f"tab_close_inv_{sel_invoice}"):
                    st.session_state["tab_ih_selected_invoice"] = None
                    try:
                        st.experimental_rerun()
                    except Exception:
                        try:
                            st.rerun()
                        except Exception:
                            pass

            try:
                csv_data = df_display.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Invoice History CSV", data=csv_data, file_name="invoice_history.csv", mime="text/csv", use_container_width=True)
            except Exception:
                pass

    # ---------------- Tab 5: Customer Master (exact copy of Customer Master page, tab-scoped) ----------------
    with tab5:
        # --- Helper: Ensure required columns exist ---
        def ensure_customer_table_columns_tab():
            try:
                conn = get_conn()
                cur = conn.cursor()
                cur.execute("PRAGMA table_info(customers)")
                cols = [r[1] for r in cur.fetchall()]
                required_cols = {"email": "TEXT", "address": "TEXT", "pan": "TEXT", "state": "TEXT"}
                for col, col_type in required_cols.items():
                    if col not in cols:
                        try:
                            cur.execute(f"ALTER TABLE customers ADD COLUMN {col} {col_type}")
                            conn.commit()
                        except Exception:
                            pass
                conn.close()
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass

        ensure_customer_table_columns_tab()

        st.markdown("""
            <div style='display:flex;align-items:center;justify-content:space-between;
                        background:#0A1F44;border-radius:10px;padding:10px 16px;margin-bottom:12px;
                        border:1px solid #0A1F44;box-shadow:0 2px 6px rgba(0,0,0,0.15);'>
                <div>
                    <h3 style='margin:0;color:#ffffff;font-weight:600;'>👤 Customer Master</h3>
                    <div style='color:rgba(255,255,255,0.8);font-size:13px;'>Manage customers with PAN & email verification</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        try:
            customers_df = fetch_customers_df() if "fetch_customers_df" in globals() else pd.DataFrame()
        except Exception:
            customers_df = pd.DataFrame()

        for col in ["name", "mobile", "email", "address", "pan", "state"]:
            if col not in customers_df.columns:
                customers_df[col] = ""

        with st.expander("➕ Add / Update Customer", expanded=False):
            c1, c2, c3 = st.columns([1.2, 1.2, 1])
            with c1:
                name = st.text_input("Name", key="cm_tab_cust_name", placeholder="Customer name")
                mobile = st.text_input("Mobile", key="cm_tab_cust_mobile", placeholder="10-digit mobile")
            with c2:
                email = st.text_input("Email", key="cm_tab_cust_email", placeholder="example@email.com")
                pan = st.text_input("PAN", key="cm_tab_cust_pan", placeholder="ABCDE1234F")
            with c3:
                address = st.text_area("Address", key="cm_tab_cust_address", placeholder="Address", height=80)
            state = st.text_input("State", key="cm_tab_cust_state", placeholder="Customer state (e.g., Delhi)")

            def valid_pan(pan_val): return bool(re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", (pan_val or "").upper()))
            def valid_email(email_val): return bool(re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", (email_val or "")))

            if pan and not valid_pan(pan):
                st.warning("⚠️ Invalid PAN format. Expected: ABCDE1234F")
            if email and not valid_email(email):
                st.warning("⚠️ Invalid email address")

            if st.button("💾 Save / Update Customer", key="cm_tab_save_cust"):
                if not name or not mobile:
                    st.error("Name and mobile are required.")
                elif pan and not valid_pan(pan):
                    st.error("Invalid PAN format.")
                elif email and not valid_email(email):
                    st.error("Invalid email address.")
                elif not state or not str(state).strip():
                    st.error("State is required for the customer.")
                else:
                    try:
                        conn = get_conn(); cur = conn.cursor()
                        try:
                            save_customer(mobile, name, "", address)
                        except Exception:
                            try:
                                cur.execute("SELECT mobile FROM customers WHERE mobile=?", (mobile,))
                                if cur.fetchone():
                                    cur.execute("UPDATE customers SET name=?, address=? WHERE mobile=?", (name, address, mobile))
                                else:
                                    cur.execute("INSERT INTO customers (mobile, name, address) VALUES (?, ?, ?)", (mobile, name, address))
                                conn.commit()
                            except Exception:
                                pass
                        try:
                            cur.execute("PRAGMA table_info(customers)")
                            cols_now = [r[1] for r in cur.fetchall()]
                            if "email" not in cols_now:
                                cur.execute("ALTER TABLE customers ADD COLUMN email TEXT")
                            if "pan" not in cols_now:
                                cur.execute("ALTER TABLE customers ADD COLUMN pan TEXT")
                            if "state" not in cols_now:
                                cur.execute("ALTER TABLE customers ADD COLUMN state TEXT")
                            conn.commit()
                        except Exception:
                            pass
                        cur.execute("UPDATE customers SET email=?, pan=?, state=? WHERE mobile=?", (email or None, (pan or "").upper() if pan else None, state or None, mobile))
                        conn.commit(); conn.close()
                        st.success(f"Customer {name} ({mobile}) saved successfully ✅")
                        try:
                            st.experimental_rerun()
                        except Exception:
                            try:
                                st.rerun()
                            except Exception:
                                pass
                    except Exception as e:
                        st.error(f"Error saving customer: {e}")

        st.markdown("<div class='cust-card'>", unsafe_allow_html=True)
        st.markdown("**📋 Customer List**")

        c1, c2 = st.columns([2, 1])
        search = c1.text_input("Search by name, mobile, PAN or state", key="cm_tab_cust_search", placeholder="Start typing...")
        limit = c2.number_input("Limit", min_value=1, max_value=500, value=50, key="cm_tab_cust_limit")

        df_display = customers_df.copy()
        if search:
            q = search.strip().lower()
            df_display = df_display[
                df_display["name"].astype(str).str.lower().str.contains(q, na=False) |
                df_display["mobile"].astype(str).str.contains(q) |
                df_display["pan"].astype(str).str.lower().str.contains(q, na=False) |
                df_display["state"].astype(str).str.lower().str.contains(q, na=False)
            ]
        df_display = df_display.head(limit)

        if df_display.empty:
            st.info("No customers found.")
        else:
            st.markdown(f"<div class='summary-pill'>👥 {len(df_display)} customers displayed</div>", unsafe_allow_html=True)
            visible_cols = [c for c in ["name", "mobile", "email", "pan", "state", "address"] if c in df_display.columns]
            st.dataframe(
                df_display[visible_cols].style.set_properties(**{
                    'color': 'white',
                    'background-color': '#1C1C1C',
                    'border-color': '#0A1F44'
                }),
                use_container_width=True,
                height=360
            )
        st.markdown("</div>", unsafe_allow_html=True)

        if not customers_df.empty:
            with st.expander("✏️ Edit Customer", expanded=False):
                sel = st.selectbox("Select customer", customers_df.apply(lambda r: f"{r['mobile']} | {r['name']}", axis=1), key="cm_tab_edit_select")
                if sel:
                    mob = sel.split("|")[0].strip()
                    conn = get_conn(); cur = conn.cursor()
                    try:
                        cur.execute("PRAGMA table_info(customers)")
                        cols = [r[1] for r in cur.fetchall()]
                        required_cols = {"email": "TEXT", "address": "TEXT", "pan": "TEXT", "state": "TEXT"}
                        for col_name, col_type in required_cols.items():
                            if col_name not in cols:
                                try:
                                    cur.execute(f"ALTER TABLE customers ADD COLUMN {col_name} {col_type}")
                                    conn.commit()
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    cur.execute("SELECT name, email, address, pan, state FROM customers WHERE mobile=?", (mob,))
                    cdata = cur.fetchone(); conn.close()

                    if cdata:
                        c1, c2 = st.columns(2)
                        new_name = c1.text_input("Name", value=cdata[0] or "", key=f"cm_tab_edit_name_{mob}")
                        new_email = c2.text_input("Email", value=cdata[1] or "", key=f"cm_tab_edit_email_{mob}")
                        new_pan = c1.text_input("PAN", value=cdata[3] or "", key=f"cm_tab_edit_pan_{mob}")
                        new_state = c2.text_input("State", value=cdata[4] or "", key=f"cm_tab_edit_state_{mob}")
                        new_address = st.text_area("Address", value=cdata[2] or "", key=f"cm_tab_edit_addr_{mob}")

                        if new_pan and not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", (new_pan or "").upper()):
                            st.warning("⚠️ Invalid PAN format")
                        if new_email and not re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", (new_email or "")):
                            st.warning("⚠️ Invalid email address")

                        if st.button("✅ Save Changes", key=f"cm_tab_save_edit_{mob}"):
                            if new_pan and not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", (new_pan or "").upper()):
                                st.error("Invalid PAN format.")
                            elif new_email and not re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", (new_email or "")):
                                st.error("Invalid email address.")
                            elif not new_state or not str(new_state).strip():
                                st.error("State is required.")
                            else:
                                try:
                                    conn = get_conn(); cur = conn.cursor()
                                    cur.execute("""
                                        UPDATE customers
                                        SET name=?, email=?, address=?, pan=?, state=?
                                        WHERE mobile=?
                                    """, (new_name, new_email, new_address, (new_pan or "").upper(), new_state, mob))
                                    conn.commit(); conn.close()
                                    st.success("Customer updated successfully ✅")
                                    try:
                                        st.experimental_rerun()
                                    except Exception:
                                        try:
                                            st.rerun()
                                        except Exception:
                                            pass
                                except Exception as e:
                                    st.error(f"Update failed: {e}")

    # ---------------- Tab 6: TCS Register (with date filters) ----------------
    with tab6:
        st.markdown("### 💰 TCS Register ")

        if df_all.empty:
            st.info("No invoices.")
        else:
            # Filter rows that have a TCS rate > 0
            df_tcs = df_all[df_all.get("tcs_rate", 0).fillna(0) > 0].copy() if "tcs_rate" in df_all.columns else pd.DataFrame()
            if df_tcs.empty:
                st.info("No TCS applicable invoices found.")
            else:
                # ---------------- Date Filter UI (same as GST Register) ----------------
                today = datetime.date.today()

                def first_day_of_prev_month(d):
                    first_this_month = d.replace(day=1)
                    prev_last = first_this_month - datetime.timedelta(days=1)
                    return prev_last.replace(day=1)

                def last_day_of_prev_month(d):
                    first_this_month = d.replace(day=1)
                    prev_last = first_this_month - datetime.timedelta(days=1)
                    return prev_last

                range_choice = st.selectbox(
                    "Range",
                    ["Current month", "Previous month", "This Financial Year", "Custom range", "All"],
                    key="tcs_reg_range",
                )

                if range_choice == "Current month":
                    start_date = today.replace(day=1)
                    end_date = today
                elif range_choice == "Previous month":
                    start_date = first_day_of_prev_month(today)
                    end_date = last_day_of_prev_month(today)
                elif range_choice == "This Financial Year":
                    if today.month >= 4:
                        start_date = datetime.date(today.year, 4, 1)
                    else:
                        start_date = datetime.date(today.year - 1, 4, 1)
                    end_date = today
                elif range_choice == "Custom range":
                    c1, c2 = st.columns(2)
                    start_date = c1.date_input("Start", value=today.replace(day=1), key="tcs_tab_start")
                    end_date = c2.date_input("End", value=today, key="tcs_tab_end")
                else:
                    start_date = None
                    end_date = None

                # Normalize invoice_date
                if "invoice_date" in df_tcs.columns:
                    try:
                        df_tcs["invoice_date"] = pd.to_datetime(df_tcs["invoice_date"], errors="coerce").dt.date
                    except Exception:
                        pass

                # Apply date filter
                if start_date is not None and end_date is not None:
                    df_tcs = df_tcs[
                        (df_tcs["invoice_date"] >= start_date) & (df_tcs["invoice_date"] <= end_date)
                    ]

                if df_tcs.empty:
                    st.info("No TCS invoices in selected range.")
                else:
                    # Keep selected columns
                    keep_cols = [
                        col
                        for col in [
                            "invoice_no",
                            "invoice_date",
                            "customer_name",
                            "customer_mobile",
                            "customer_pan",
                            "tcs_rate",
                            "tcs_amount",
                            "grand_total",
                        ]
                        if col in df_tcs.columns
                    ]

                    df_display = df_tcs[keep_cols].fillna("")

                    # Summary
                    total_tcs = df_display["tcs_amount"].astype(float).sum()
                    st.markdown(
                        f"<div class='summary-pill'>🧾 {len(df_display)} invoices with TCS • Total TCS ₹{total_tcs:,.2f}</div>",
                        unsafe_allow_html=True,
                    )

                    # Show table
                    st.dataframe(df_display, height=380, use_container_width=True)

                    # Download button
                    try:
                        st.download_button(
                            "⬇️ Download TCS Register CSV",
                            data=df_display.to_csv(index=False).encode("utf-8"),
                            file_name="tcs_register.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    except Exception:
                        pass


    # ---------------- Tab 7: SFT Register ----------------
    with tab7:
        st.markdown("### 🏦 SFT Register (cash sales > ₹200,000)")
        if df_all.empty:
            st.info("No invoices.")
        else:
            df_cash = df_all[df_all.get("payment_mode","").astype(str).str.lower() == "cash"].copy() if "payment_mode" in df_all.columns else pd.DataFrame()
            if df_cash.empty:
                st.info("No cash receipts found.")
            else:
                df_cash["invoice_date_parsed"] = pd.to_datetime(df_cash["invoice_date"], errors="coerce")
                def fy_label(d):
                    if pd.isna(d):
                        return ""
                    y = d.year
                    return f"{y}-{y+1}" if d.month >= 4 else f"{y-1}-{y}"
                df_cash["fy"] = df_cash["invoice_date_parsed"].apply(fy_label)
                df_summary = df_cash.groupby(["customer_name","customer_mobile","customer_pan","fy"], dropna=False).agg(total_cash=("payment_received","sum"), invoices=("invoice_no","count")).reset_index()
                df_sft = df_summary[df_summary["total_cash"] > 200000]
                if df_sft.empty:
                    st.success("No customers exceeded ₹2L in cash receipts.")
                else:
                    st.dataframe(df_sft.fillna(""), height=360, use_container_width=True)
                    try:
                        st.download_button("⬇️ Download SFT CSV", data=df_sft.to_csv(index=False).encode("utf-8"), file_name="sft_register.csv", use_container_width=True)
                    except Exception:
                        pass

    # ---------------- Tab 8: GST Register ----------------
    with tab8:
        st.markdown("### 🧾 GST Register ")

        if df_all.empty:
            st.info("No invoices.")
        else:
            # only GST invoices
            df_gst = df_all[df_all.get("gst_rate", 0).fillna(0) > 0].copy() if "gst_rate" in df_all.columns else pd.DataFrame()
            if df_gst.empty:
                st.info("No GST invoices.")
            else:
                # --- date range filter UI ---
                today = datetime.date.today()

                def first_day_of_prev_month(d):
                    first_this_month = d.replace(day=1)
                    prev_last = first_this_month - datetime.timedelta(days=1)
                    return prev_last.replace(day=1)

                def last_day_of_prev_month(d):
                    first_this_month = d.replace(day=1)
                    prev_last = first_this_month - datetime.timedelta(days=1)
                    return prev_last

                range_choice = st.selectbox(
                    "Range",
                    ["Current month", "Previous month", "This Financial Year", "Custom range", "All"],
                    key="gst_reg_range",
                )

                if range_choice == "Current month":
                    start_date = today.replace(day=1)
                    end_date = today
                elif range_choice == "Previous month":
                    start_date = first_day_of_prev_month(today)
                    end_date = last_day_of_prev_month(today)
                elif range_choice == "This Financial Year":
                    if today.month >= 4:
                        start_date = datetime.date(today.year, 4, 1)
                    else:
                        start_date = datetime.date(today.year - 1, 4, 1)
                    end_date = today
                elif range_choice == "Custom range":
                    c1, c2 = st.columns(2)
                    start_date = c1.date_input("Start", value=today.replace(day=1), key="gst_tab_start")
                    end_date = c2.date_input("End", value=today, key="gst_tab_end")
                else:
                    start_date = None
                    end_date = None

                # Resolve company state
                company_state = ""
                try:
                    if "fetch_company" in globals():
                        comp = fetch_company()
                        if isinstance(comp, dict):
                            company_state = str(comp.get("state", "") or "").strip()
                        elif isinstance(comp, (list, tuple)) and len(comp) >= 6:
                            company_state = str(comp[5] or "").strip()
                    if not company_state:
                        try:
                            conn_cs = get_conn()
                            cur_cs = conn_cs.cursor()
                            cur_cs.execute("SELECT state FROM company LIMIT 1")
                            crow = cur_cs.fetchone()
                            if crow and len(crow) > 0:
                                company_state = str(crow[0] or "").strip()
                        except Exception:
                            pass
                        finally:
                            try:
                                conn_cs.close()
                            except Exception:
                                pass
                except Exception:
                    company_state = ""
                company_state_norm = (company_state or "").strip().lower()

                # Resolve customer state
                try:
                    custs = fetch_customers_df() if "fetch_customers_df" in globals() else pd.DataFrame()
                    if not custs.empty and "mobile" in custs.columns and "state" in custs.columns:
                        state_map = dict(zip(custs["mobile"].astype(str), custs["state"].astype(str)))
                        df_gst["customer_state_resolved"] = df_gst.apply(
                            lambda r: (r.get("customer_state") or state_map.get(str(r.get("customer_mobile")) or "", "")),
                            axis=1,
                        )
                    else:
                        df_gst["customer_state_resolved"] = df_gst.get("customer_state", "")
                except Exception:
                    df_gst["customer_state_resolved"] = df_gst.get("customer_state", "")

                # Normalize dates
                if "invoice_date" in df_gst.columns:
                    try:
                        df_gst["invoice_date"] = pd.to_datetime(df_gst["invoice_date"], errors="coerce").dt.date
                    except Exception:
                        pass

                # Apply date filter
                if start_date is not None and end_date is not None:
                    df_gst = df_gst[
                        (df_gst["invoice_date"] >= start_date) & (df_gst["invoice_date"] <= end_date)
                    ]

                if df_gst.empty:
                    st.info("No GST invoices in selected range.")
                else:
                    # Build GST Register
                    gst_rows = []
                    for _, r in df_gst.iterrows():
                        gst_amt = float(r.get("gst_amount") or 0.0)
                        subtotal = float(r.get("subtotal") or 0.0)
                        invoice_gst_type = (r.get("gst_type") or "").strip().lower()
                        cust_state = str(r.get("customer_state_resolved") or "").strip().lower()

                        # Determine intra/inter
                        if company_state_norm:
                            is_intra = cust_state == company_state_norm
                        else:
                            is_intra = invoice_gst_type.startswith("intra")

                        if is_intra:
                            cgst = gst_amt / 2
                            sgst = gst_amt / 2
                            igst = 0
                            gst_type_label = "Intra-State"
                        else:
                            cgst = 0
                            sgst = 0
                            igst = gst_amt
                            gst_type_label = "Inter-State"

                        gst_rows.append(
                            {
                                "Invoice No": r.get("invoice_no"),
                                "Invoice Date": r.get("invoice_date"),
                                "Customer": r.get("customer_name"),
                                "Customer Mobile": r.get("customer_mobile"),
                                "Customer State": r.get("customer_state_resolved"),
                                "GSTIN": r.get("customer_gstin"),
                                "Taxable Value": subtotal,
                                "GST Rate (%)": r.get("gst_rate"),
                                "GST Type": gst_type_label,
                                "CGST": round(cgst, 2),
                                "SGST": round(sgst, 2),
                                "IGST": round(igst, 2),
                                "Total GST": round(gst_amt, 2),
                                "Grand Total": r.get("grand_total"),
                            }
                        )

                    df_gstr = pd.DataFrame(gst_rows)
                    st.markdown(
                        f"<div class='summary-pill'>🧾 {len(df_gstr)} GST invoices • Total GST ₹{df_gstr['Total GST'].sum():,.2f}</div>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(df_gstr.fillna(""), height=320, use_container_width=True)
                    st.download_button(
                        "⬇️ Download GST Register CSV",
                        data=df_gstr.to_csv(index=False).encode("utf-8"),
                        file_name="gst_register.csv",
                        use_container_width=True,
                    )

                    # ---------------- Category-wise HSN Summary ----------------
                    with st.expander("📊 Category-wise HSN Summary (B2B vs B2C)", expanded=False):
                        if "safe_float" not in globals():
                            def safe_float(v, default=0.0):
                                try:
                                    return float(v)
                                except Exception:
                                    return default

                        FORCED_HSN_LOCAL = globals().get("FORCED_HSN", "7301")

                        item_rows = []
                        for _, inv in df_gst.iterrows():
                            inv_meta = {
                                "invoice_no": inv.get("invoice_no"),
                                "invoice_date": inv.get("invoice_date"),
                                "customer_name": inv.get("customer_name"),
                                "customer_mobile": inv.get("customer_mobile"),
                                "customer_gstin": inv.get("customer_gstin"),
                                "subtotal": inv.get("subtotal") or 0.0,
                                "gst_amount": inv.get("gst_amount") or 0.0,
                                "gst_rate": inv.get("gst_rate") or 0.0,
                            }
                            items_json_str = inv.get("items_json") if "items_json" in inv else None
                            try:
                                items = json.loads(items_json_str) if items_json_str else []
                            except Exception:
                                items = []

                            if items:
                                for it in items:
                                    category = (
                                        it.get("category")
                                        or it.get("cat")
                                        or it.get("item_category")
                                        or "Uncategorized"
                                    )
                                    hsn = (
                                        it.get("hsn")
                                        or it.get("hsn_code")
                                        or it.get("hsnCode")
                                        or FORCED_HSN_LOCAL
                                    )
                                    qty = safe_float(it.get("qty") or it.get("quantity"))
                                    rate = safe_float(it.get("rate") or it.get("price"))
                                    making = safe_float(it.get("making"))
                                    amount = safe_float(it.get("amount") or (qty * rate + making))
                                    item_rows.append(
                                        {
                                            **inv_meta,
                                            "hsn": str(hsn or FORCED_HSN_LOCAL),
                                            "category": str(category or "Uncategorized"),
                                            "qty": qty,
                                            "rate": rate,
                                            "making": making,
                                            "amount": amount,
                                        }
                                    )
                            else:
                                item_rows.append(
                                    {
                                        **inv_meta,
                                        "hsn": FORCED_HSN_LOCAL,
                                        "category": "Uncategorized",
                                        "qty": 1,
                                        "rate": inv_meta["subtotal"],
                                        "making": 0,
                                        "amount": inv_meta["subtotal"],
                                    }
                                )

                        df_items_gst = pd.DataFrame(item_rows)
                        if df_items_gst.empty:
                            st.info("No item-level data available.")
                        else:
                            df_items_gst["customer_gstin"] = df_items_gst["customer_gstin"].fillna("")
                            df_items_gst["B2B_B2C"] = df_items_gst["customer_gstin"].apply(
                                lambda x: "B2B" if str(x).strip() else "B2C"
                            )

                            df_items_gst["gst_amount_est"] = df_items_gst.apply(
                                lambda r: safe_float(r.get("amount")) * safe_float(r.get("gst_rate")) / 100,
                                axis=1,
                            )

                            df_cat_hsn_summary = (
                                df_items_gst.groupby(["category", "hsn", "B2B_B2C"], dropna=False)
                                .agg(
                                    taxable_value=("amount", "sum"),
                                    gst_value=("gst_amount_est", "sum"),
                                    invoice_count=("invoice_no", "nunique"),
                                    total_qty=("qty", "sum"),
                                )
                                .reset_index()
                            )

                            df_cat_hsn_summary[["taxable_value", "gst_value", "total_qty"]] = df_cat_hsn_summary[
                                ["taxable_value", "gst_value", "total_qty"]
                            ].round(2)

                            view_choice = st.radio(
                                "View summary by:",
                                ["Category + HSN + B2B/B2C", "Category + B2B/B2C (HSN summed)"],
                                horizontal=True,
                                key="cat_hsn_view",
                            )

                            if view_choice == "Category + B2B/B2C (HSN summed)":
                                df_view = (
                                    df_cat_hsn_summary.groupby(["category", "B2B_B2C"], dropna=False)
                                    .agg(
                                        taxable_value=("taxable_value", "sum"),
                                        gst_value=("gst_value", "sum"),
                                        invoice_count=("invoice_count", "sum"),
                                        total_qty=("total_qty", "sum"),
                                    )
                                    .reset_index()
                                )
                            else:
                                df_view = df_cat_hsn_summary.copy()

                            # ✅ Safe sorting (prevents KeyError: 'hsn')
                            sort_cols = [c for c in ["category", "hsn", "B2B_B2C"] if c in df_view.columns]
                            try:
                                df_display = df_view.sort_values(sort_cols)
                            except Exception:
                                df_display = df_view

                            st.dataframe(df_display.fillna(""), height=380, use_container_width=True)

                            st.download_button(
                                "⬇️ Download Category-HSN Summary CSV",
                                data=df_display.to_csv(index=False).encode("utf-8"),
                                file_name="gst_category_hsn_summary.csv",
                                use_container_width=True,
                            )


    # ---------------- Tab 9: Cash Register ----------------
    with tab9:
        st.markdown("### 💵 Cash Register (compact, hideable)")
        with st.expander("📊 Show / Hide Cash Register", expanded=False):
            c1, c2 = st.columns([1,1])
            today = datetime.date.today()
            start_date = c1.date_input("Start Date", value=today.replace(day=1), key="cash_tab_start")
            end_date = c2.date_input("End Date", value=today, key="cash_tab_end")

            try:
                conn = get_conn(); cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS cash_deposits (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        deposit_date TEXT,
                        amount REAL,
                        note TEXT,
                        created_at TEXT
                    )
                """)
                conn.commit(); conn.close()
            except Exception:
                pass

            try:
                conn = get_conn()
                df_open_coll = pd.read_sql_query("SELECT COALESCE(SUM(amount),0) AS total_collected FROM payments WHERE lower(COALESCE(mode,''))='cash' AND date < ?", conn, params=(str(start_date),))
                df_open_dep = pd.read_sql_query("SELECT COALESCE(SUM(amount),0) AS total_deposited FROM cash_deposits WHERE deposit_date < ?", conn, params=(str(start_date),))
                conn.close()
                opening = float(df_open_coll.iloc[0,0]) - float(df_open_dep.iloc[0,0])
            except Exception:
                opening = 0.0

            try:
                conn = get_conn()
                df_pay = pd.read_sql_query("""
                    SELECT date, COALESCE(SUM(amount),0) AS collected
                    FROM payments WHERE lower(COALESCE(mode,''))='cash' AND date BETWEEN ? AND ?
                    GROUP BY date ORDER BY date
                """, conn, params=(str(start_date), str(end_date)))
                df_depo = pd.read_sql_query("SELECT deposit_date AS date, COALESCE(SUM(amount),0) AS deposited FROM cash_deposits WHERE deposit_date BETWEEN ? AND ? GROUP BY deposit_date ORDER BY deposit_date", conn, params=(str(start_date), str(end_date)))
                conn.close()
            except Exception:
                df_pay = pd.DataFrame()
                df_depo = pd.DataFrame()

            days = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date)}).assign(date=lambda d: d["date"].dt.date)
            if not df_pay.empty:
                df_pay["date"] = pd.to_datetime(df_pay["date"]).dt.date
            if not df_depo.empty:
                df_depo["date"] = pd.to_datetime(df_depo["date"]).dt.date
            days = days.merge(df_pay, on="date", how="left").merge(df_depo, on="date", how="left")
            days["collected"] = days["collected"].fillna(0.0)
            days["deposited"] = days["deposited"].fillna(0.0)
            days["cash_on_hand"] = days["collected"].cumsum() + opening - days["deposited"].cumsum()
            display_df = days.rename(columns={"date":"Date","collected":"Collected","deposited":"Deposited","cash_on_hand":"Cash on Hand"})
            st.dataframe(display_df[["Date","Collected","Deposited","Cash on Hand"]].fillna(0.0), height=360, use_container_width=True)

            d1, d2, d3 = st.columns([1.2,1,1])
            dep_date = d1.date_input("Deposit Date", value=today, key="cash_dep_date")
            dep_amt = d2.number_input("Amount (₹)", min_value=0.0, step=0.01, key="cash_dep_amt")
            dep_note = d3.text_input("Note", key="cash_dep_note")
            if st.button("Record Deposit", key="cash_record"):
                if dep_amt <= 0:
                    st.error("Enter positive amount.")
                else:
                    try:
                        conn = get_conn(); cur = conn.cursor()
                        cur.execute("INSERT INTO cash_deposits (deposit_date, amount, note, created_at) VALUES (?, ?, ?, ?)", (str(dep_date), float(dep_amt), dep_note or None, str(datetime.datetime.now())))
                        conn.commit(); conn.close()
                        st.success("Deposit recorded.")
                        try:
                            st.rerun()
                        except Exception:
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass
                    except Exception as e:
                        st.error(f"Could not record deposit: {e}")

    # ---------------- Tab 10: Payments Ledger ----------------
    with tab10:
        import pandas as _pd
        st.markdown("<div class='crm-card'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                    <h4 style='margin:0;color:#ffffff'>💳 Payments Ledger</h4>
                    <div style='color:#d1d5db;font-size:13px;margin-top:4px'>Transactions & reconciliation (compact view)</div>
                </div>
                <div style='text-align:right'>
                    <div class='summary-pill'>Loading payments...</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Ensure payments.customer_name exists (safe migration)
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='payments'")
            if cur.fetchone():
                cur.execute("PRAGMA table_info(payments)")
                cols = [r[1] for r in cur.fetchall()]
                if "customer_name" not in cols:
                    cur.execute("ALTER TABLE payments ADD COLUMN customer_name TEXT")
                    conn.commit()
            conn.close()
        except Exception:
            try: conn.close()
            except: pass

        try:
            conn = get_conn()
            df_pay = pd.read_sql_query("SELECT * FROM payments ORDER BY created_at DESC LIMIT 1000", conn)
        except Exception as e:
            st.error(f"Could not fetch payments: {e}")
            df_pay = _pd.DataFrame()
        finally:
            try:
                conn.close()
            except Exception:
                pass

        for c in ["id", "invoice_no", "customer_name", "customer_mobile", "amount", "mode", "date", "note", "created_at"]:
            if c not in df_pay.columns:
                df_pay[c] = ""

        st.markdown("<div class='mini-row'>", unsafe_allow_html=True)
        if not df_pay.empty:
            latest5 = df_pay.head(5)
            row_html = "<div style='display:flex;gap:10px;flex-wrap:wrap;'>"
            for _, r in latest5.iterrows():
                amt = f"₹{(float(r.get('amount') or 0)):,.2f}"
                who = r.get('customer_name') or r.get('customer_mobile') or '—'
                label = f"{r.get('invoice_no') or '—'} • {who} • {amt}"
                row_html += f"<div class='mini-item' style='padding:6px 8px;border-radius:6px;background:#161616;border:1px solid #2b2b2b;color:#e5e7eb;font-weight:600;margin-right:6px'>{label}</div>"
            row_html += "</div>"
            st.markdown(row_html, unsafe_allow_html=True)
        else:
            st.info("No payments recorded yet.")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔍 Show Filters & Full Ledger", expanded=False):
            f1, f2, f3 = st.columns([2,1,1])
            search = f1.text_input("Search name / mobile / invoice / mode", placeholder="Type to filter...", key="pl_search")
            min_amt = f2.number_input("Min amount (₹)", min_value=0.0, value=0.0, step=1.0, key="pl_min_amt")
            limit = f3.number_input("Rows to show", min_value=10, max_value=1000, value=200, step=10, key="pl_limit")

            df_display = df_pay.copy()
            if search and search.strip():
                q = search.strip().lower()
                df_display = df_display[
                    df_display["customer_name"].astype(str).str.lower().str.contains(q, na=False)
                    | df_display["customer_mobile"].astype(str).str.contains(q)
                    | df_display["invoice_no"].astype(str).str.contains(q)
                    | df_display["mode"].astype(str).str.lower().str.contains(q, na=False)
                ]
            if min_amt and min_amt > 0:
                df_display["amount_num"] = pd.to_numeric(df_display["amount"], errors="coerce").fillna(0)
                df_display = df_display[df_display["amount_num"] >= float(min_amt)]
                df_display.drop(columns=["amount_num"], inplace=True)
            df_display = df_display.head(int(limit))
            tot = df_display["amount"].apply(lambda v: float(v) if str(v).strip() else 0.0).sum() if not df_display.empty else 0
            st.markdown(f"<div class='summary-pill'>Showing: {len(df_display)} • Total ₹{tot:,.2f}</div>", unsafe_allow_html=True)
            show_cols = [c for c in ["id", "date", "invoice_no", "customer_name", "customer_mobile", "amount", "mode", "note"] if c in df_display.columns]
            st.dataframe(df_display[show_cols].style.format({"amount": "₹{:,.2f}"}), height=380)

            a1, a2 = st.columns([1,1])
            if a1.button("Export CSV"):
                csvb = df_display.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csvb, file_name="payments_ledger_export.csv", mime="text/csv")
            if a2.button("Refresh"):
                st.rerun()

        with st.expander("✏️ Edit Payment (select row)", expanded=False):
            options = []
            id_to_row = {}
            for _, r in df_pay.iterrows():
                pid = r.get("id")
                if pd.isna(pid):
                    continue
                label = f"{int(pid)} • {r.get('invoice_no') or '—'} • {r.get('customer_name') or r.get('customer_mobile') or '—'} • ₹{(float(r.get('amount') or 0)):,.2f}"
                options.append(label)
                id_to_row[label] = r.to_dict()

            if not options:
                st.info("No payments available to edit.")
            else:
                sel = st.selectbox("Choose payment to edit", options, key="pl_edit_selector")
                prow = id_to_row.get(sel)
                pid = int(prow.get("id"))
                invoice_no = prow.get("invoice_no") or ""
                customer_mobile = prow.get("customer_mobile") or ""
                customer_name_val = prow.get("customer_name") or ""
                amount_val = float(prow.get("amount") or 0.0)
                try:
                    pdate_val = pd.to_datetime(prow.get("date")).date() if prow.get("date") else datetime.date.today()
                except Exception:
                    pdate_val = datetime.date.today()
                mode_val = prow.get("mode") or "--Select--"
                note_val = prow.get("note") or ""

                e1, e2, e3 = st.columns([1.2,1.2,1])
                with e1:
                    invoice_no = st.text_input("Invoice no (blank if advance)", value=invoice_no, key=f"pl_inv_sel_{pid}")
                    customer_mobile = st.text_input("Customer mobile", value=customer_mobile, key=f"pl_mob_sel_{pid}")
                    customer_name_val = st.text_input("Customer name", value=customer_name_val, key=f"pl_name_sel_{pid}")
                with e2:
                    amount_val = st.number_input("Amount (₹)", value=amount_val, step=0.01, key=f"pl_amt_sel_{pid}")
                    pdate_val = st.date_input("Date", value=pdate_val, key=f"pl_date_sel_{pid}")
                with e3:
                    mode_val = st.selectbox("Mode", ["--Select--","Cash","Card","UPI","Bank Transfer","Cheque","Other"],
                                            index=0 if mode_val=="--Select--" else 1, key=f"pl_mode_sel_{pid}")
                    note_val = st.text_input("Note", value=note_val, key=f"pl_note_sel_{pid}")

                if st.button("💾 Save changes", key=f"pl_save_sel_{pid}"):
                    try:
                        conn = get_conn()
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE payments
                            SET invoice_no=?, customer_mobile=?, customer_name=?, amount=?, date=?, mode=?, note=?
                            WHERE id=?
                        """, (invoice_no or None, customer_mobile or None, customer_name_val or None,
                              float(amount_val), str(pdate_val),
                              (mode_val if mode_val!="--Select--" else None), note_val, pid))
                        conn.commit()
                        conn.close()
                        st.success(f"Payment #{pid} updated successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not update payment: {e}")

        with st.expander("➕ Quick Add Payment", expanded=False):
            a1, a2, a3, a4 = st.columns([1.2, 1, 1, 1])
            with a1:
                new_cust_name = st.text_input("Customer Name", key="pl_new_cust_name", placeholder="Customer full name")
            with a2:
                new_cust_mobile = st.text_input("Customer Mobile", key="pl_new_cust_mobile", placeholder="10-digit mobile")
            find_col, dummy = st.columns([1, 2])
            with find_col:
                if st.button("🔎 Find invoices", key="pl_find_invoices"):
                    st.session_state["pl_search_name"] = (new_cust_name or "").strip()
                    st.session_state["pl_search_mobile"] = (new_cust_mobile or "").strip()

            search_name = st.session_state.get("pl_search_name", "").strip()
            search_mobile = st.session_state.get("pl_search_mobile", "").strip()

            invoice_options = []
            invoice_map = {}
            if search_name or search_mobile:
                try:
                    conn = get_conn(); cur = conn.cursor()
                    params = []

                    sql = "SELECT invoice_no, date, customer_name, customer_mobile, COALESCE(grand_total,0) as grand_total FROM invoices WHERE 1=0"
                    if search_mobile:
                        sql += " OR customer_mobile = ?"
                        params.append(search_mobile)
                    if search_name:
                        sql += " OR lower(customer_name) LIKE ?"
                        params.append(f"%{search_name.lower()}%")
                    sql += " ORDER BY date DESC LIMIT 200"
                    cur.execute(sql, tuple(params))
                    rows = cur.fetchall()
                    for r in rows:
                        inv_no = r[0]; inv_date = r[1] or ""; inv_name = r[2] or ""; inv_mob = r[3] or ""; inv_total = float(r[4] or 0.0)
                        cur.execute("SELECT COALESCE(SUM(amount),0) FROM payments WHERE invoice_no = ?", (inv_no,))
                        paid = cur.fetchone()[0] or 0.0
                        outstanding = max(0.0, float(inv_total) - float(paid))
                        label = f"{inv_no} • {inv_date} • {inv_name or inv_mob} • Outstanding: ₹{outstanding:,.2f}"
                        invoice_options.append(label)
                        invoice_map[label] = {
                            "invoice_no": inv_no,
                            "date": inv_date,
                            "customer_name": inv_name,
                            "customer_mobile": inv_mob,
                            "grand_total": inv_total,
                            "paid": paid,
                            "outstanding": outstanding,
                        }
                    conn.close()
                except Exception as e:
                    st.error(f"Could not lookup invoices: {e}")

            selected_invoice_label = None
            selected_invoice = None
            if invoice_options:
                selected_invoice_label = st.selectbox("Select invoice to apply payment", ["--Select invoice--"] + invoice_options, key="pl_selected_invoice_label")
                if selected_invoice_label and selected_invoice_label != "--Select invoice--":
                    selected_invoice = invoice_map.get(selected_invoice_label)

            with a3:
                default_amt = float(selected_invoice.get("outstanding", 0.0)) if selected_invoice else 0.0
                prev_amt = st.session_state.get("pl_new_amt", None)
                if prev_amt is None:
                    amt_value = default_amt
                else:
                    if st.session_state.get("pl_selected_invoice_label") != selected_invoice_label:
                        amt_value = default_amt
                    else:
                        amt_value = prev_amt
                new_amt = st.number_input("Amount (₹)", min_value=0.0, step=0.01, value=float(amt_value), key="pl_new_amt")
            with a4:
                new_mode = st.selectbox("Mode", ["Cash","Card","UPI","Bank Transfer","Cheque","Other"], key="pl_new_mode")
            new_note = st.text_input("Note / Invoice", key="pl_new_note", placeholder="Payment note / reference")

            if st.button("Save Payment (Quick)", key="pl_save_quick"):
                if not (new_cust_name or new_cust_mobile or selected_invoice):
                    st.error("Enter customer name or mobile OR select an invoice from search results.")
                elif new_amt <= 0:
                    st.error("Enter a positive amount.")
                else:
                    try:
                        conn = get_conn(); cur = conn.cursor()
                        inv_no_to_save = selected_invoice.get("invoice_no") if selected_invoice else None
                        name_to_save = (selected_invoice.get("customer_name") or new_cust_name).strip() if (selected_invoice and selected_invoice.get("customer_name")) else (new_cust_name.strip() or None)
                        mob_to_save = (selected_invoice.get("customer_mobile") or new_cust_mobile).strip() if (selected_invoice and selected_invoice.get("customer_mobile")) else (new_cust_mobile.strip() or None)
                        try:
                            cur.execute("PRAGMA table_info(payments)")
                            pcols = [r[1] for r in cur.fetchall()]
                            if "customer_name" not in pcols:
                                cur.execute("ALTER TABLE payments ADD COLUMN customer_name TEXT")
                                conn.commit()
                        except Exception:
                            pass
                        cur.execute("""
                            INSERT INTO payments (date, invoice_no, customer_name, customer_mobile, amount, mode, note, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (str(datetime.date.today()), inv_no_to_save, name_to_save, mob_to_save, float(new_amt), new_mode, new_note, str(datetime.datetime.now())))
                        conn.commit()
                        conn.close()
                        st.success("Payment recorded successfully.")
                        st.session_state.pop("pl_search_name", None)
                        st.session_state.pop("pl_search_mobile", None)
                        st.session_state.pop("pl_selected_invoice_label", None)
                        st.session_state.pop("pl_new_amt", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not save payment: {e}")



# ---------------- Stock Master (4 Tabs: Overview | Add/Edit | FIFO | Movement with Filters) ----------------
elif page in ("Stock Master", "📦 Stock Master"):
    import pandas as pd
    import datetime

    # --- Header ---
    st.markdown("""
        <div style='display:flex;align-items:center;justify-content:space-between;
                    background:#0A1F44;border-radius:10px;padding:10px 16px;margin-bottom:12px;
                    border:1px solid #0A1F44;box-shadow:0 2px 6px rgba(0,0,0,0.15);'>
            <div>
                <h3 style='margin:0;color:#ffffff;font-weight:600;'>📦 Stock Master</h3>
                <div style='color:rgba(255,255,255,0.8);font-size:13px;'>Manage stock, FIFO valuation & movement register (FY-based)</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- Theme ---
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] { background-color:#1C1C1C; color:#fff; font-family:"Inter","Segoe UI",sans-serif; }
        [data-testid="stDataFrame"] table { background-color:#1C1C1C !important; color:#fff !important; border-collapse:collapse !important; font-size:13px; }
        [data-testid="stDataFrame"] th { background-color:#0A1F44 !important; color:#fff !important; }
        [data-testid="stDataFrame"] tr:nth-child(even) td { background-color:#262626 !important; }
        [data-testid="stDataFrame"] tr:hover td { background-color:#0A1F44 !important; }
        div[data-testid="stButton"] > button { background:#0A1F44; color:white; border:none; border-radius:6px; padding:4px 10px; font-size:13px; }
        div[data-testid="stButton"] > button:hover { background-color:#132a5c; }
        </style>
    """, unsafe_allow_html=True)

    # --- Ensure DB schema ---
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(stocks)")
    cols = [r[1] for r in cur.fetchall()]
    if "total_value" not in cols:
        cur.execute("ALTER TABLE stocks ADD COLUMN total_value REAL DEFAULT 0")
    if "avg_rate" not in cols:
        cur.execute("ALTER TABLE stocks ADD COLUMN avg_rate REAL DEFAULT 0")
    cur.execute("PRAGMA table_info(stock_transactions)")
    tx_cols = [r[1] for r in cur.fetchall()]
    if "rate" not in tx_cols:
        cur.execute("ALTER TABLE stock_transactions ADD COLUMN rate REAL")
    if "value" not in tx_cols:
        cur.execute("ALTER TABLE stock_transactions ADD COLUMN value REAL")
    conn.commit(); conn.close()

    # --- Fetch Stock Master ---
    stocks_df = fetch_stocks_df()
    if "total_value" not in stocks_df.columns:
        stocks_df["total_value"] = 0.0
    if "avg_rate" not in stocks_df.columns:
        stocks_df["avg_rate"] = 0.0

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Stock Overview", "➕ Add / Edit Stock", "📈 FIFO Valuation", "📜 Stock Movement Register"
    ])

    # ---------------- TAB 1: STOCK OVERVIEW ----------------
    with tab1:
        st.markdown("### 📊 Stock Overview")
        if stocks_df.empty:
            st.info("No stock records.")
        else:
            # compute weighted avg rate from transactions
            conn = get_conn()
            df_tx = pd.read_sql_query("""
                SELECT stock_id,
                       SUM(CASE WHEN change>0 THEN change ELSE 0 END) AS total_in,
                       SUM(CASE WHEN change>0 THEN change*rate ELSE 0 END) AS total_value_in
                FROM stock_transactions GROUP BY stock_id
            """, conn)
            conn.close()
            df_tx["avg_rate_calc"] = df_tx.apply(
                lambda r: (r["total_value_in"]/r["total_in"]) if r["total_in"]>0 else 0, axis=1
            )
            stocks_df = stocks_df.merge(
                df_tx[["stock_id","avg_rate_calc"]],
                left_on="id", right_on="stock_id", how="left"
            )
            stocks_df["avg_rate_final"] = stocks_df["avg_rate_calc"].fillna(stocks_df["avg_rate"])
            stocks_df["total_value"] = stocks_df["quantity"].astype(float)*stocks_df["avg_rate_final"].astype(float)
            st.markdown(f"**💰 Total Stock Value: ₹{stocks_df['total_value'].sum():,.2f}**")
            st.dataframe(
                stocks_df[["id","category","purity","description","unit","quantity","avg_rate_final","total_value"]]
                .rename(columns={"avg_rate_final":"Avg Rate"})
                .style.format({"quantity":"{:.2f}","Avg Rate":"₹{:.2f}","total_value":"₹{:.2f}"})
            )

    # ---------------- TAB 2: ADD / EDIT STOCK ----------------
    with tab2:
        st.markdown("### ➕ Add / Edit Stock")

        # ---- Add Section ----
        st.subheader("Add / Update Stock Entry")
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        with c1:
            category = st.selectbox("Category", ["Gold Ornaments", "Silver Ornaments", "Diamond Ornaments"], key="add_cat")
            purity = st.text_input("Purity", value="22K" if category=="Gold Ornaments" else "Std", key="add_purity")
            unit = "Grms" if category != "Diamond Ornaments" else "Ct"
        with c2:
            description = st.text_input("Description", key="add_desc", placeholder="Item name")
            qty = st.number_input("Qty (±)", value=0.0, step=0.01, key="add_qty")
        with c3:
            rate = st.number_input("Rate (₹/unit)", value=0.0, step=0.01, key="add_rate")
            tx_date = st.date_input("Date", value=datetime.date.today(), key="add_txdate")
        with c4:
            reason = st.text_input("Reason", value="Opening", key="add_reason")

        if st.button("💾 Save / Update Stock", key="add_save_stock"):
            if not description:
                st.error("Enter description.")
            else:
                try:
                    conn = get_conn(); cur = conn.cursor()
                    cur.execute("SELECT id, quantity, avg_rate FROM stocks WHERE description=? AND category=? AND purity=?",
                                (description, category, purity))
                    existing = cur.fetchone()
                    if existing:
                        sid, old_qty, old_rate = existing
                        new_qty = float(old_qty or 0) + float(qty)
                        new_avg_rate = (
                            ((old_qty * old_rate) + (qty * rate)) / new_qty if new_qty > 0 else rate
                        )
                        cur.execute("""
                            UPDATE stocks SET quantity=?, avg_rate=?, total_value=?, created_at=? WHERE id=?
                        """, (new_qty, new_avg_rate, new_qty * new_avg_rate, str(datetime.datetime.now()), sid))
                        stock_id = sid
                    else:
                        new_avg_rate = rate
                        cur.execute("""
                            INSERT INTO stocks (category, purity, description, unit, quantity, avg_rate, total_value, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (category, purity, description, unit, qty, new_avg_rate, qty * new_avg_rate, str(datetime.datetime.now())))
                        stock_id = cur.lastrowid

                    cur.execute("""
                        INSERT INTO stock_transactions (stock_id, tx_date, change, rate, value, reason, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (stock_id, str(tx_date), qty, rate, qty * rate, reason, str(datetime.datetime.now())))
                    conn.commit(); conn.close()
                    st.success(f"✅ Stock updated (ID {stock_id}) → Qty {qty} @ ₹{rate}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating stock: {e}")

        st.divider()
        st.subheader("✏️ Edit Existing Stock")

        if stocks_df.empty:
            st.info("No stock to edit.")
        else:
            sel = st.selectbox("Select item to edit", stocks_df.apply(lambda r: f"{r['id']} | {r['description']} | Qty:{r['quantity']:.2f}", axis=1))
            sid = int(sel.split("|")[0])
            record = stocks_df[stocks_df["id"] == sid].iloc[0]

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                new_qty = st.number_input("Set Quantity", value=float(record["quantity"]), step=0.01)
            with e2:
                new_rate = st.number_input("Rate (₹)", value=float(record["avg_rate_final"]), step=0.01)
            with e3:
                reason = st.text_input("Reason", value="Manual Adjustment")
            with e4:
                tx_date = st.date_input("Edit Date", value=datetime.date.today())

            delta_qty = float(new_qty) - float(record["quantity"])

            if st.button("💾 Apply Edit", key=f"edit_apply_{sid}"):
                try:
                    conn = get_conn(); cur = conn.cursor()
                    cur.execute("""
                        UPDATE stocks
                        SET quantity=?, avg_rate=?, total_value=?, created_at=?
                        WHERE id=?
                    """, (new_qty, new_rate, new_qty * new_rate, str(datetime.datetime.now()), sid))
                    cur.execute("""
                        INSERT INTO stock_transactions (stock_id, tx_date, change, rate, value, reason, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (sid, str(tx_date), delta_qty, new_rate, delta_qty * new_rate, reason, str(datetime.datetime.now())))
                    conn.commit(); conn.close()
                    st.success("✅ Stock record updated successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating stock: {e}")

    # ---------------- TAB 3: FIFO VALUATION ----------------
    with tab3:
        st.markdown("### 📈 FIFO Valuation (Financial Year Based)")
        today = datetime.date.today()
        fy_start = datetime.date(today.year if today.month>=4 else today.year-1,4,1)
        fy_end = datetime.date(today.year+1 if today.month>=4 else today.year,3,31)
        range_mode = st.selectbox("Select Range", ["This Financial Year", "Previous Financial Year", "Custom Range"])
        if range_mode == "This Financial Year":
            start_date, end_date = fy_start, today
        elif range_mode == "Previous Financial Year":
            start_date = datetime.date(fy_start.year-1,4,1)
            end_date = datetime.date(fy_start.year,3,31)
        else:
            c1, c2 = st.columns(2)
            start_date = c1.date_input("Start", fy_start)
            end_date = c2.date_input("End", today)

        conn = get_conn()
        df_tx = pd.read_sql_query("""
            SELECT stock_id, tx_date, change, COALESCE(rate,0) as rate
            FROM stock_transactions
            WHERE date(tx_date) BETWEEN ? AND ?
            ORDER BY stock_id, date(tx_date), id
        """, conn, params=(str(start_date), str(end_date)))
        conn.close()

        if df_tx.empty:
            st.info("No transactions in this range.")
        else:
            df_tx["tx_date"] = pd.to_datetime(df_tx["tx_date"]).dt.date
            fifo_data = []
            for sid, grp in df_tx.groupby("stock_id"):
                stack = []
                for _, row in grp.iterrows():
                    ch, r = row["change"], row["rate"]
                    if ch > 0:
                        stack.append([ch, r])
                    elif ch < 0:
                        out = -ch
                        while out > 0 and stack:
                            q, rate = stack[0]
                            use = min(out, q)
                            q -= use
                            out -= use
                            if q <= 0:
                                stack.pop(0)
                            else:
                                stack[0][0] = q
                qty = sum(q for q, _ in stack)
                val = sum(q * r for q, r in stack)
                fifo_data.append({"stock_id": sid, "fifo_qty": qty, "fifo_value": val, "fifo_rate": (val/qty if qty>0 else 0)})
            df_fifo = pd.DataFrame(fifo_data)
            df_fifo = df_fifo.merge(stocks_df[["id","description","category","purity","unit"]],
                                    left_on="stock_id", right_on="id", how="left")
            st.markdown(f"**Total FIFO Valuation: ₹{df_fifo['fifo_value'].sum():,.2f}**")
            st.dataframe(df_fifo[["description","category","purity","unit","fifo_qty","fifo_rate","fifo_value"]]
                         .style.format({"fifo_qty":"{:.2f}","fifo_rate":"₹{:.2f}","fifo_value":"₹{:.2f}"}))

    # ---------------- TAB 4: STOCK MOVEMENT REGISTER ----------------
    with tab4:
        st.markdown("### 📜 Stock Movement Register")

        # --- Filters (Category + Date) ---
        c1, c2, c3 = st.columns([1.5,1,1])
        with c1:
            category_filter = st.selectbox(
                "Category", ["All","Gold Ornaments","Silver Ornaments","Diamond Ornaments"]
            )
        with c2:
            range_choice = st.selectbox("Range", ["Last 30 Days","This Month","This Financial Year","Custom"])
        with c3:
            today = datetime.date.today()
            if range_choice == "This Month":
                start_date = today.replace(day=1)
            elif range_choice == "This Financial Year":
                start_date = datetime.date(today.year if today.month>=4 else today.year-1,4,1)
            elif range_choice == "Last 30 Days":
                start_date = today - datetime.timedelta(days=30)
            else:
                start_date = st.date_input("Start Date", today - datetime.timedelta(days=30))
            end_date = st.date_input("End Date", today)

        conn = get_conn()
        df_move = pd.read_sql_query("""
            SELECT t.stock_id, s.description, s.category, s.purity, s.unit,
                   t.tx_date, t.change, COALESCE(t.rate,0) as rate, COALESCE(t.value,0) as value, t.reason
            FROM stock_transactions t
            LEFT JOIN stocks s ON t.stock_id = s.id
            WHERE date(t.tx_date) BETWEEN ? AND ?
            ORDER BY date(t.tx_date) DESC, t.id DESC
        """, conn, params=(str(start_date), str(end_date)))
        conn.close()

        if df_move.empty:
            st.info("No stock movements found in this period.")
        else:
            if category_filter != "All":
                df_move = df_move[df_move["category"] == category_filter]
            st.markdown(f"**📅 Showing {len(df_move)} records from {start_date} → {end_date}**")
            st.dataframe(
                df_move.style.format({"change":"{:.2f}","rate":"₹{:.2f}","value":"₹{:.2f}"}), height=420
            )
            csv = df_move.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Movements CSV",
                data=csv,
                file_name=f"stock_movement_{start_date}_{end_date}.csv",
                mime="text/csv"
            )

# ---------------- Payments Ledger (Compact + Hidden Sections, CRM Theme) ----------------
elif page in ("Payments Ledger", "💳 Payments Ledger"):
    import pandas as pd
    import datetime

    # --- Theme CSS (compact) ---
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] { background-color: #1C1C1C; color: #FFFFFF; font-family: "Inter", "Segoe UI", sans-serif; }
        .crm-card { background: #2a2a2a; border: 1px solid #0A1F44; border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; }
        .summary-pill { background: #0A1F44; color: #fff; padding: 6px 10px; border-radius: 8px; font-size: 13px; font-weight:600; display:inline-block; margin-bottom:8px; }
        .mini-row { display:flex; gap:10px; align-items:center; color:#d1d5db; font-size:13px; margin-bottom:6px; }
        .mini-item { background:#161616; padding:6px 8px; border-radius:6px; border:1px solid #2b2b2b; color:#e5e7eb; font-weight:600; }
        div[data-testid="stButton"] > button { background:#0A1F44 !important; color:#fff !important; border-radius:6px !important; padding:4px 8px !important; font-weight:600 !important; }
        [data-testid="stDataFrame"] table { background-color:#1C1C1C !important; color:#fff !important; font-size:13px; }
        [data-testid="stDataFrame"] th { background-color:#0A1F44 !important; color:#fff !important; }
        [data-testid="stDataFrame"] tr:nth-child(even) td { background-color:#262626 !important; }
        </style>
    """, unsafe_allow_html=True)

    # --- Ensure payments.customer_name exists (safe migration) ---
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='payments'")
        if cur.fetchone():
            cur.execute("PRAGMA table_info(payments)")
            cols = [r[1] for r in cur.fetchall()]
            if "customer_name" not in cols:
                cur.execute("ALTER TABLE payments ADD COLUMN customer_name TEXT")
                conn.commit()
        conn.close()
    except Exception:
        try: conn.close()
        except: pass

    # --- Load payments (safe) ---
    conn = get_conn()
    try:
        df_pay = pd.read_sql_query("SELECT * FROM payments ORDER BY created_at DESC LIMIT 1000", conn)
    except Exception as e:
        st.error(f"Could not fetch payments: {e}")
        df_pay = pd.DataFrame()
    finally:
        conn.close()

    # ensure common cols exist to avoid KeyError
    for c in ["id", "invoice_no", "customer_name", "customer_mobile", "amount", "mode", "date", "note", "created_at"]:
        if c not in df_pay.columns:
            df_pay[c] = ""

    # --- Compact header + quick summary ---
    st.markdown("<div class='crm-card'>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center'>
            <div>
                <h4 style='margin:0;color:#ffffff'>💳 Payments Ledger</h4>
                <div style='color:#d1d5db;font-size:13px;margin-top:4px'>Transactions & reconciliation (compact view)</div>
            </div>
            <div style='text-align:right'>
                <div class='summary-pill'>Total records: {len(df_pay)}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- Show tiny recent 5 strip ---
    if not df_pay.empty:
        latest5 = df_pay.head(5)
        row_html = "<div class='mini-row'>"
        for _, r in latest5.iterrows():
            amt = f"₹{safe_float(r.get('amount',0)):,.2f}"
            who = r.get('customer_name') or r.get('customer_mobile') or '—'
            label = f"{r.get('invoice_no') or '—'} • {who} • {amt}"
            row_html += f"<div class='mini-item'>{label}</div>"
        row_html += "</div>"
        st.markdown(row_html, unsafe_allow_html=True)
    else:
        st.info("No payments recorded yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Hidden: Filters & Full Ledger ----------------
    with st.expander("🔍 Show Filters & Full Ledger", expanded=False):
        f1, f2, f3 = st.columns([2, 1, 1])
        search = f1.text_input("Search name / mobile / invoice / mode", placeholder="Type to filter...", key="pl_search")
        min_amt = f2.number_input("Min amount (₹)", min_value=0.0, value=0.0, step=1.0, key="pl_min_amt")
        limit = f3.number_input("Rows to show", min_value=10, max_value=1000, value=200, step=10, key="pl_limit")

        df_display = df_pay.copy()
        if search and search.strip():
            q = search.strip().lower()
            df_display = df_display[
                df_display["customer_name"].astype(str).str.lower().str.contains(q, na=False)
                | df_display["customer_mobile"].astype(str).str.contains(q)
                | df_display["invoice_no"].astype(str).str.contains(q)
                | df_display["mode"].astype(str).str.lower().str.contains(q, na=False)
            ]
        if min_amt and min_amt > 0:
            df_display["amount_num"] = pd.to_numeric(df_display["amount"], errors="coerce").fillna(0)
            df_display = df_display[df_display["amount_num"] >= float(min_amt)]
            df_display.drop(columns=["amount_num"], inplace=True)
        df_display = df_display.head(int(limit))
        tot = df_display["amount"].apply(safe_float).sum() if not df_display.empty else 0
        st.markdown(f"<div class='summary-pill'>Showing: {len(df_display)} • Total ₹{tot:,.2f}</div>", unsafe_allow_html=True)
        show_cols = [c for c in ["id", "date", "invoice_no", "customer_name", "customer_mobile", "amount", "mode", "note"] if c in df_display.columns]
        st.dataframe(df_display[show_cols].style.format({"amount": "₹{:,.2f}"}), height=380)

        a1, a2 = st.columns([1,1])
        if a1.button("Export CSV"):
            csvb = df_display.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csvb, file_name="payments_ledger_export.csv", mime="text/csv")
        if a2.button("Refresh"):
            st.rerun()

    # ---------------- Hidden: Edit Payment (dropdown selector) ----------------
    with st.expander("✏️ Edit Payment (select row)", expanded=False):
        # Build selector options: "id • invoice_no • name • mobile • ₹amount"
        options = []
        id_to_row = {}
        for _, r in df_pay.iterrows():
            pid = r.get("id")
            if pd.isna(pid): 
                continue
            label = f"{int(pid)} • {r.get('invoice_no') or '—'} • {r.get('customer_name') or r.get('customer_mobile') or '—'} • ₹{safe_float(r.get('amount',0)):,.2f}"
            options.append(label)
            id_to_row[label] = r.to_dict()

        if not options:
            st.info("No payments available to edit.")
        else:
            sel = st.selectbox("Choose payment to edit", options, key="pl_edit_selector")
            prow = id_to_row.get(sel)
            # populate form with selected row
            pid = int(prow.get("id"))
            invoice_no = prow.get("invoice_no") or ""
            customer_mobile = prow.get("customer_mobile") or ""
            customer_name_val = prow.get("customer_name") or ""
            amount_val = safe_float(prow.get("amount",0.0))
            try:
                pdate_val = pd.to_datetime(prow.get("date")).date() if prow.get("date") else datetime.date.today()
            except Exception:
                pdate_val = datetime.date.today()
            mode_val = prow.get("mode") or "--Select--"
            note_val = prow.get("note") or ""

            e1, e2, e3 = st.columns([1.2,1.2,1])
            with e1:
                invoice_no = st.text_input("Invoice no (blank if advance)", value=invoice_no, key=f"pl_inv_sel_{pid}")
                customer_mobile = st.text_input("Customer mobile", value=customer_mobile, key=f"pl_mob_sel_{pid}")
                customer_name_val = st.text_input("Customer name", value=customer_name_val, key=f"pl_name_sel_{pid}")
            with e2:
                amount_val = st.number_input("Amount (₹)", value=amount_val, step=0.01, key=f"pl_amt_sel_{pid}")
                pdate_val = st.date_input("Date", value=pdate_val, key=f"pl_date_sel_{pid}")
            with e3:
                mode_val = st.selectbox("Mode", ["--Select--","Cash","Card","UPI","Bank Transfer","Cheque","Other"],
                                        index=0 if mode_val=="--Select--" else 1, key=f"pl_mode_sel_{pid}")
                note_val = st.text_input("Note", value=note_val, key=f"pl_note_sel_{pid}")

            if st.button("💾 Save changes", key=f"pl_save_sel_{pid}"):
                try:
                    conn = get_conn()
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE payments
                        SET invoice_no=?, customer_mobile=?, customer_name=?, amount=?, date=?, mode=?, note=?
                        WHERE id=?
                    """, (invoice_no or None, customer_mobile or None, customer_name_val or None,
                          float(amount_val), str(pdate_val),
                          (mode_val if mode_val!="--Select--" else None), note_val, pid))
                    conn.commit()
                    conn.close()
                    st.success(f"Payment #{pid} updated successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not update payment: {e}")

    # ---------------- Hidden: Quick Add Payment (with invoice lookup) ----------------
    with st.expander("➕ Quick Add Payment", expanded=False):
        a1, a2, a3, a4 = st.columns([1.2, 1, 1, 1])
        with a1:
            new_cust_name = st.text_input("Customer Name", key="pl_new_cust_name", placeholder="Customer full name")
        with a2:
            new_cust_mobile = st.text_input("Customer Mobile", key="pl_new_cust_mobile", placeholder="10-digit mobile")
        # find invoices button
        find_col, dummy = st.columns([1, 2])
        with find_col:
            if st.button("🔎 Find invoices", key="pl_find_invoices"):
                # Save search terms to session so selection persists
                st.session_state["pl_search_name"] = (new_cust_name or "").strip()
                st.session_state["pl_search_mobile"] = (new_cust_mobile or "").strip()

        # load search terms (persistent)
        search_name = st.session_state.get("pl_search_name", "").strip()
        search_mobile = st.session_state.get("pl_search_mobile", "").strip()

        # prepare invoice matches list if any search provided
        invoice_options = []
        invoice_map = {}  # label -> dict(row)
        if search_name or search_mobile:
            try:
                conn = get_conn(); cur = conn.cursor()
                # fetch candidate invoices by exact mobile OR name LIKE
                params = []
                sql = "SELECT invoice_no, date, customer_name, customer_mobile, COALESCE(grand_total,0) as grand_total FROM invoices WHERE 1=0"
                if search_mobile:
                    sql += " OR customer_mobile = ?"
                    params.append(search_mobile)
                if search_name:
                    sql += " OR lower(customer_name) LIKE ?"
                    params.append(f"%{search_name.lower()}%")
                sql += " ORDER BY date DESC LIMIT 200"
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
                # build label and compute outstanding
                for r in rows:
                    inv_no = r[0]
                    inv_date = r[1] or ""
                    inv_name = r[2] or ""
                    inv_mob = r[3] or ""
                    inv_total = safe_float(r[4], 0.0)
                    # compute paid so far for this invoice
                    cur.execute("SELECT COALESCE(SUM(amount),0) FROM payments WHERE invoice_no = ?", (inv_no,))
                    paid = cur.fetchone()[0] or 0.0
                    outstanding = max(0.0, float(inv_total) - safe_float(paid, 0.0))
                    label = f"{inv_no} • {inv_date} • {inv_name or inv_mob} • Outstanding: ₹{outstanding:,.2f}"
                    invoice_options.append(label)
                    invoice_map[label] = {
                        "invoice_no": inv_no,
                        "date": inv_date,
                        "customer_name": inv_name,
                        "customer_mobile": inv_mob,
                        "grand_total": inv_total,
                        "paid": paid,
                        "outstanding": outstanding,
                    }
                conn.close()
            except Exception as e:
                st.error(f"Could not lookup invoices: {e}")
                try: conn.close()
                except: pass

        # Show invoice selector if any matches
        selected_invoice_label = None
        selected_invoice = None
        if invoice_options:
            selected_invoice_label = st.selectbox("Select invoice to apply payment", ["--Select invoice--"] + invoice_options, key="pl_selected_invoice_label")
            if selected_invoice_label and selected_invoice_label != "--Select invoice--":
                selected_invoice = invoice_map.get(selected_invoice_label)

        # Amount and mode fields (prefill amount with outstanding if invoice selected)
        with a3:
            default_amt = safe_float(selected_invoice.get("outstanding", 0.0)) if selected_invoice else 0.0
            # if user already typed amount earlier, preserve it
            prev_amt = st.session_state.get("pl_new_amt", None)
            if prev_amt is None:
                amt_value = default_amt
            else:
                # if invoice changed, default to outstanding; otherwise keep previous typed value
                if st.session_state.get("pl_selected_invoice_label") != selected_invoice_label:
                    amt_value = default_amt
                else:
                    amt_value = prev_amt
            new_amt = st.number_input("Amount (₹)", min_value=0.0, step=0.01, value=float(amt_value), key="pl_new_amt")
        with a4:
            new_mode = st.selectbox("Mode", ["Cash","Card","UPI","Bank Transfer","Cheque","Other"], key="pl_new_mode")
        new_note = st.text_input("Note / Invoice", key="pl_new_note", placeholder="Payment note / reference")

        # Save payment: if invoice selected, save invoice_no on payment row
        if st.button("Save Payment (Quick)", key="pl_save_quick"):
            if not (new_cust_name or new_cust_mobile or selected_invoice):
                st.error("Enter customer name or mobile OR select an invoice from search results.")
            elif new_amt <= 0:
                st.error("Enter a positive amount.")
            else:
                try:
                    conn = get_conn(); cur = conn.cursor()
                    inv_no_to_save = selected_invoice.get("invoice_no") if selected_invoice else None
                    name_to_save = (selected_invoice.get("customer_name") or new_cust_name).strip() if (selected_invoice and selected_invoice.get("customer_name")) else (new_cust_name.strip() or None)
                    mob_to_save = (selected_invoice.get("customer_mobile") or new_cust_mobile).strip() if (selected_invoice and selected_invoice.get("customer_mobile")) else (new_cust_mobile.strip() or None)

                    # ensure payments table has customer_name column (safe)
                    try:
                        cur.execute("PRAGMA table_info(payments)")
                        pcols = [r[1] for r in cur.fetchall()]
                        if "customer_name" not in pcols:
                            cur.execute("ALTER TABLE payments ADD COLUMN customer_name TEXT")
                            conn.commit()
                    except Exception:
                        pass

                    cur.execute("""
                        INSERT INTO payments (date, invoice_no, customer_name, customer_mobile, amount, mode, note, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (str(datetime.date.today()), inv_no_to_save, name_to_save, mob_to_save, float(new_amt), new_mode, new_note, str(datetime.datetime.now())))
                    conn.commit()
                    conn.close()
                    st.success("Payment recorded successfully.")
                    # clear search & inputs so user can see updated ledger
                    st.session_state.pop("pl_search_name", None)
                    st.session_state.pop("pl_search_mobile", None)
                    st.session_state.pop("pl_selected_invoice_label", None)
                    st.session_state.pop("pl_new_amt", None)
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not save payment: {e}")



# ---------------- Company Settings (CRM-styled) with GSTIN <-> State validation ----------------
elif page in ("Company Settings", "🏢 Company Settings"):
    import io
    import base64
    import sqlite3

    # --- Page header (CRM style) ---
    st.markdown("""
        <div style='background:#0A1F44;border-radius:10px;padding:10px 18px;
                    margin-bottom:12px;border:1px solid #0A1F44;box-shadow:0 2px 6px rgba(0,0,0,0.15);'>
            <div>
                <h3 style='margin:0;color:#ffffff;font-weight:600;'>🏢 Company Profile & Settings</h3>
                <div style='color:rgba(255,255,255,0.8);font-size:13px;'>Update company details, logo, signature, and mandatory state (with GSTIN validation)</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- Theme CSS ---
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #1C1C1C;
            color: #FFFFFF;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .crm-card {
            background: #2a2a2a;
            border: 1px solid #0A1F44;
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }
        .summary-pill {
            background: #0A1F44;
            color: #fff;
            padding: 6px 10px;
            border-radius: 8px;
            font-size: 13px;
            font-weight:600;
            display:inline-block;
            margin-bottom:10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- GST State Code Map ---
    GST_STATE_CODES = {
        "Jammu & Kashmir": 1, "Himachal Pradesh": 2, "Punjab": 3, "Chandigarh": 4, "Uttarakhand": 5,
        "Haryana": 6, "Delhi": 7, "Rajasthan": 8, "Uttar Pradesh": 9, "Bihar": 10, "Sikkim": 11,
        "Arunachal Pradesh": 12, "Nagaland": 13, "Manipur": 14, "Mizoram": 15, "Tripura": 16,
        "Meghalaya": 17, "Assam": 18, "West Bengal": 19, "Jharkhand": 20, "Odisha": 21,
        "Chhattisgarh": 22, "Madhya Pradesh": 23, "Gujarat": 24, "Daman and Diu": 25,
        "Dadra & Nagar Haveli": 26, "Maharashtra": 27, "Andhra Pradesh (Old)": 28, "Karnataka": 29,
        "Goa": 30, "Lakshadweep": 31, "Kerala": 32, "Tamil Nadu": 33, "Puducherry": 34,
        "Andaman & Nicobar Islands": 35, "Telangana": 36, "Andhra Pradesh": 37, "Ladakh": 38,
        "Other/Union Territory": None
    }

    def _map_state_to_code(state_name):
        if not state_name:
            return None
        for k, v in GST_STATE_CODES.items():
            if k.lower() == state_name.lower():
                return v
        return None

    # --- Ensure "state" column exists ---
    def ensure_state_column():
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('company','companies')")
            tables = [r[0] for r in cur.fetchall()]
            for tbl in tables:
                cur.execute(f"PRAGMA table_info({tbl})")
                cols = [r[1] for r in cur.fetchall()]
                if "state" not in cols:
                    cur.execute(f"ALTER TABLE {tbl} ADD COLUMN state TEXT")
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

    ensure_state_column()

    # --- Load company details ---
    try:
        company = fetch_company()
    except Exception:
        company = None

    comp_name = company[0] if company and len(company) > 0 else ""
    comp_gstin = company[1] if company and len(company) > 1 else ""
    comp_addr = company[2] if company and len(company) > 2 else ""
    comp_logo = company[3] if company and len(company) > 3 else None
    comp_sig = company[4] if company and len(company) > 4 else None
    comp_state = ""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT state FROM company LIMIT 1")
        row = cur.fetchone()
        if row:
            comp_state = row[0] or ""
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    st.markdown(f"<div class='summary-pill'>🏷️ Company: <b>{comp_name or 'Not set'}</b></div>", unsafe_allow_html=True)

    # --- Edit Section ---
    with st.expander("✏️ Edit Company Details", expanded=True):
        st.markdown("<div class='crm-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            name = st.text_input("Company Name", value=comp_name)
            gstin = st.text_input("Company GSTIN", value=comp_gstin)
            addr = st.text_area("Address", value=comp_addr, height=100)

            states = list(GST_STATE_CODES.keys())
            default_idx = states.index(comp_state) if comp_state in states else (len(states)-1)
            state = st.selectbox("State (mandatory)", states, index=default_idx)

        with col2:
            logo_file = st.file_uploader("Logo", type=["png", "jpg", "jpeg"])
            sig_file = st.file_uploader("Signature", type=["png", "jpg", "jpeg"])

            if logo_file:
                st.image(logo_file.read(), caption="Logo Preview", width=150)
            if sig_file:
                st.image(sig_file.read(), caption="Signature Preview", width=150)

        if st.button("💾 Save Company"):
            if not name.strip():
                st.error("Company name is required.")
                st.stop()
            if not state.strip():
                st.error("State is required.")
                st.stop()

            # GSTIN vs State Validation
            gst_code = _map_state_to_code(state)
            if gstin.strip():
                try:
                    gst_prefix = int(gstin[:2])
                    if gst_code and gst_prefix != gst_code:
                        st.error(f"GSTIN prefix ({gst_prefix}) does not match state '{state}' (expected {gst_code:02d}).")
                        st.stop()
                except Exception:
                    st.warning("Invalid GSTIN format; skipping validation.")

            # Save
            try:
                save_company(name.strip(), gstin.strip(), addr.strip(), None, None)
            except Exception as e:
                st.warning(f"save_company() fallback: {e}")

            try:
                conn = get_conn()
                cur = conn.cursor()
                cur.execute("UPDATE company SET state=? WHERE ROWID IN (SELECT ROWID FROM company LIMIT 1)", (state,))
                conn.commit()
                st.success("✅ Company updated successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Error saving to database: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        st.markdown("</div>", unsafe_allow_html=True)
