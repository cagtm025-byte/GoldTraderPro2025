
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
        "customer_gstin":"TEXT","customer_address":"TEXT","gst_rate":"REAL DEFAULT 0","gst_type":"TEXT",
        "subtotal":"REAL DEFAULT 0","cgst":"REAL DEFAULT 0","sgst":"REAL DEFAULT 0","igst":"REAL DEFAULT 0",
        "gst_total":"REAL DEFAULT 0","grand_total":"REAL DEFAULT 0","status":"TEXT DEFAULT 'Active'",
        "cancelled_at":"TEXT","payment_status":"TEXT DEFAULT 'Unpaid'","payment_mode":"TEXT",
        "payment_received":"REAL DEFAULT 0","payment_date":"TEXT"
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

def insert_invoice_and_items(invoice_no_requested, date_str, customer_mobile, customer_name, customer_gstin, customer_address, gst_rate, gst_type, items, payment_status="Unpaid", payment_mode=None, payment_received=None, payment_date=None, payment_is_advance=False, payment_note=None):
    conn = get_conn(); cur = conn.cursor()
    # stock availability check
    for it in items:
        sid = it.get("stock_id")
        if sid not in (None,"","None"):
            cur.execute("SELECT quantity FROM stocks WHERE id=?", (int(sid),))
            r = cur.fetchone()
            if not r: raise Exception(f"Stock id {sid} not found")
            avail = safe_float(r[0],0.0)
            if safe_float(it.get("qty",0.0)) > avail + 1e-9:
                raise Exception(f"Insufficient stock for '{it.get('item_name')}' (avail {avail})")
    subtotal = sum([safe_float(x.get("amount",0.0)) for x in items])
    gst_total = subtotal * (safe_float(gst_rate,0.0)/100.0)
    if (gst_type or "").lower()=="intra-state": cgst=sgst=gst_total/2.0; igst=0.0
    else: cgst=sgst=0.0; igst=gst_total
    grand_total = subtotal + gst_total
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
    invoice_payload = {"invoice_no":actual_invoice_no,"date":date_str,"customer_mobile":customer_mobile,"customer_name":customer_name,"customer_gstin":customer_gstin,"customer_address":customer_address,"gst_rate":safe_float(gst_rate,0.0),"gst_type":gst_type,"subtotal":subtotal,"cgst":cgst,"sgst":sgst,"igst":igst,"gst_total":gst_total,"grand_total":grand_total,"status":"Active","payment_status":payment_status,"payment_mode":payment_mode,"payment_received":safe_float(payment_received,0.0),"payment_date":payment_date}
    insert_dict_dynamic(conn,"invoices",invoice_payload)
    for it in items:
        item_payload={"invoice_no":actual_invoice_no,"stock_id":it.get("stock_id"),"category":it.get("category"),"purity":it.get("purity"),"hsn":FORCED_HSN,"item_name":it.get("item_name"),"qty":safe_float(it.get("qty",0.0)),"unit":it.get("unit"),"rate":safe_float(it.get("rate",0.0)),"making":safe_float(it.get("making",0.0)),"amount":safe_float(it.get("amount",0.0))}
        insert_dict_dynamic(conn,"invoice_items",item_payload)
        if item_payload.get("stock_id") not in (None,"","None"):
            sid=int(item_payload.get("stock_id"))
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
    if safe_float(payment_received,0.0)>0:
        if payment_is_advance:
            conn.commit(); conn.close()
            create_advance_note(customer_mobile=customer_mobile, amount=payment_received, date=payment_date or str(datetime.date.today()), mode=payment_mode, note=payment_note)
            conn = get_conn()
        else:
            add_payment(invoice_no=actual_invoice_no, amount=payment_received, date=payment_date or str(datetime.date.today()), mode=payment_mode, note=payment_note, is_advance=False, customer_mobile=customer_mobile, conn=conn)
    conn.commit(); conn.close(); return actual_invoice_no

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
    if invoice_row and (invoice_row.get("gst_type") or "").lower()=="intra-state":
        cgst = gst_total/2.0; sgst = gst_total/2.0; igst = 0.0
    else:
        cgst = 0.0; sgst = 0.0; igst = gst_total
    grand_total = float(invoice_row.get("grand_total", subtotal + gst_total) if invoice_row else subtotal + gst_total)

    totals_x = left + usable_w*0.45
    ty = (table_top - h_tbl) - 20
    c.setFont(BASE_FONT_NAME, 9); c.setFillColor(colors.HexColor("#4d4d4d"))
    c.drawRightString(right - 8, ty, f"Subtotal: {_fmt_val(subtotal)}"); ty -= 14
    if cgst and sgst:
        c.drawRightString(right - 8, ty, f"CGST: {_fmt_val(cgst)}"); ty -= 14
        c.drawRightString(right - 8, ty, f"SGST: {_fmt_val(sgst)}"); ty -= 16
    else:
        c.drawRightString(right - 8, ty, f"IGST: {_fmt_val(igst)}"); ty -= 18
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

# ---------------- UI: sidebar, header & pages (kept exactly as Final.py) ----------------
# CSS & sidebar header
st.markdown("""
<style>
.sidebar-sep {height:1px;background:#e6e9ee;margin:12px 0;}
.header-card{
  display:flex;
  gap:18px;
  align-items:center;
  padding:18px;
  border-radius:12px;
  background: linear-gradient(90deg, #ffffff, #f8fafc);
  box-shadow: 0 6px 18px rgba(3,7,18,0.04);
  margin-bottom:10px;
}
.logo-circle{
  width:64px;height:64px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:26px;background:linear-gradient(135deg,#fde68a,#f59e0b);color:#082032;font-weight:800;
}
.title-block{ display:flex; flex-direction:column; }
.title-main{ font-size:22px; color:#052a56; font-weight:900; margin:0; padding:0; line-height:1; }
.title-sub{ font-size:13px; color:#475569; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

top_buttons = [ ("📊","Dashboard"), ("🧾","Create Invoice"), ("🛒","Sales Register"), ("💳","Payments Ledger"), ("📦","Stock Master") ]
bottom_buttons = [ ("🏢","Company Settings"), ("👥","Customer Master"), ("📚","Invoice History"), ("💳","Advances"), ("🧾","Outstanding Summary") ]

st.sidebar.markdown(f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'><div style='font-size:25px;'>💠</div><div><div style='font-size:20px;color:#DAA520;font-weight:800;'>GoldTrader Pro</div><div style='font-size:20px;color:#6b7280;'></div></div></div>", unsafe_allow_html=True)

for (emoji,label) in top_buttons:
    if st.sidebar.button(f"{emoji} {label}", key=f"btn_{label}"):
        st.session_state["page"] = label

st.sidebar.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)

for (emoji,label) in bottom_buttons:
    if st.sidebar.button(f"{emoji} {label}", key=f"btn_{label}2"):
        st.session_state["page"] = label

st.sidebar.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)

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

st.markdown("<div class='header-card'>"
            "<div class='logo-circle'>GT</div>"
            "<div class='title-block'>"
            "<div class='title-main'>GoldTrader Pro</div>"
            "<div class='title-sub'>Invoicing & Inventory — simple, secure, professional</div>"
            "</div>"
            "</div>", unsafe_allow_html=True)

page = st.session_state["page"]

# ---------------- Pages (Dashboard, Create Invoice, Sales Register, Stock Master, Advances, etc.)
# The pages' implementations are kept from your Final.py (unchanged logic) and will use the tenant-aware DB functions above.

# Dashboard
if page == "Dashboard":
    company = fetch_company()
    company_name = company[0] if company and company[0] else "GoldTrader Pro"
    st.markdown(f"## Welcome! — {company_name}")

    st.markdown("---")
    st.header("Sales & Finance Snapshot")

    conn = get_conn()
    df_inv = pd.read_sql_query("SELECT invoice_no,date,COALESCE(grand_total,0) AS grand_total FROM invoices", conn)
    df_pay = pd.read_sql_query("SELECT invoice_no,COALESCE(amount,0) AS amount, date, COALESCE(is_advance,0) AS is_advance FROM payments", conn)
    stocks_df = pd.read_sql_query("SELECT * FROM stocks", conn)
    conn.close()

    today = datetime.date.today(); last_30 = today - datetime.timedelta(days=30)
    def sum_in_range(df, date_col, start, end, val_col):
        if df.empty: return 0.0
        try:
            df[date_col] = pd.to_datetime(df[date_col]).dt.date
        except:
            pass
        sel = df[(df[date_col] >= start) & (df[date_col] <= end)]
        return float(sel[val_col].sum()) if not sel.empty else 0.0

    sales_today = sum_in_range(df_inv,"date",today,today,"grand_total")
    sales_30 = sum_in_range(df_inv,"date",last_30,today,"grand_total")
    payments_30 = sum_in_range(df_pay,"date",last_30,today,"amount")

    c1,c2,c3,c4 = st.columns(4, gap="large")
    c1.markdown(f"<div style='background: linear-gradient(135deg,#fff7ed,#ffedd5); padding:14px; border-radius:12px;'><div style='font-size:13px;color:#92400e;font-weight:700'>Sales (Today)</div><div style='font-size:24px;color:#92400e;font-weight:800'>₹{sales_today:,.2f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div style='background: linear-gradient(135deg,#ecfccb,#bbf7d0); padding:14px; border-radius:12px;'><div style='font-size:13px;color:#065f46;font-weight:700'>Sales (Last 30 days)</div><div style='font-size:24px;color:#065f46;font-weight:800'>₹{sales_30:,.2f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div style='background: linear-gradient(135deg,#e0f2fe,#bfdbfe); padding:14px; border-radius:12px;'><div style='font-size:13px;color:#0f172a;font-weight:700'>Payments (Last 30 days)</div><div style='font-size:24px;color:#0f172a;font-weight:800'>₹{payments_30:,.2f}</div></div>", unsafe_allow_html=True)

    conn = get_conn()
    total_outstanding_df = pd.read_sql_query("""
        SELECT inv.customer_mobile, inv.customer_name, COALESCE(inv.grand_total,0) AS grand_total, COALESCE(applied.applied_sum,0) AS applied
        FROM invoices inv
        LEFT JOIN (
            SELECT invoice_no, COALESCE(SUM(amount),0) AS applied_sum FROM payments WHERE COALESCE(is_advance,0)=0 GROUP BY invoice_no
        ) applied ON applied.invoice_no = inv.invoice_no
        WHERE COALESCE(inv.status,'Active') != 'Cancelled'
    """, conn)
    conn.close()
    total_outstanding = float((total_outstanding_df["grand_total"] - total_outstanding_df["applied"]).sum()) if not total_outstanding_df.empty else 0.0
    c4.markdown(f"<div style='background: linear-gradient(135deg,#fce7f3,#e9d5ff); padding:14px; border-radius:12px;'><div style='font-size:13px;color:#5b21b6;font-weight:700'>Total Outstanding</div><div style='font-size:24px;color:#5b21b6;font-weight:800'>₹{total_outstanding:,.2f}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    left_col, right_col = st.columns([2,1], gap="large")
    with left_col:
        st.subheader("Sales — Daily Performance (30 days)")
        if df_inv.empty:
            st.info("No sales data.")
        else:
            df_inv["date"] = pd.to_datetime(df_inv["date"]).dt.date
            df_daily = df_inv[(df_inv["date"] >= last_30) & (df_inv["date"] <= today)].groupby("date", as_index=False).agg(total_amount=pd.NamedAgg("grand_total","sum"))
            idx = pd.date_range(start=last_30, end=today)
            df_daily_all = pd.DataFrame({"date": idx.date})
            df_daily = pd.merge(df_daily_all, df_daily, how="left", on="date").fillna(0)
            if df_daily["total_amount"].sum() == 0: st.info("No sales recorded in the last 30 days.")
            else:
                chart = alt.Chart(df_daily).mark_area().encode(x=alt.X('date:T', title='Date'), y=alt.Y('total_amount:Q', title='Sales (₹)'), tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('total_amount:Q', title='Sales')])
                st.altair_chart(chart.properties(height=300), use_container_width=True)
    with right_col:
        st.subheader("Inventory — Available by Category")
        if stocks_df.empty: st.info("No stock records found.")
        else:
            stocks_grouped = stocks_df.groupby('category', as_index=False).agg(total_qty=pd.NamedAgg('quantity','sum')).sort_values('total_qty', ascending=False)
            stocks_grouped['total_qty'] = stocks_grouped['total_qty'].astype(float)
            st.table(stocks_grouped.rename(columns={'category':'Category','total_qty':'Available Qty'}).assign(**{'Available Qty':stocks_grouped['total_qty'].map('{:,.2f}'.format)}))
            bchart = alt.Chart(stocks_grouped).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(x=alt.X('total_qty:Q', title='Available Qty'), y=alt.Y('category:N', sort='-x', title='Category'), tooltip=[alt.Tooltip('category:N', title='Category'), alt.Tooltip('total_qty:Q', title='Available Qty', format=',.2f')]).properties(height=320)
            st.altair_chart(bchart, use_container_width=True)

# ---------------- Advances ----------------
elif page == "Advances" or page == "💳 Advances":
    st.title("Customer Advances")
    st.subheader("Create New Advance Note")
    customers_df = fetch_customers_df()
    cust_options = ["--Select Customer--"] + (customers_df["mobile"].astype(str).tolist() if not customers_df.empty else [])
    selected_customer = st.selectbox("Customer mobile", cust_options, key="adv_select_customer_main")
    adv_customer_mobile = None
    adv_customer_name = ""
    if selected_customer and selected_customer != "--Select Customer--":
        adv_customer_mobile = selected_customer
        c = customers_df[customers_df["mobile"] == selected_customer].iloc[0]
        adv_customer_name = c["name"]
        st.markdown(f"**Customer name:** {adv_customer_name}")
    else:
        adv_customer_mobile = st.text_input("Or enter mobile for new customer", key="adv_manual_mobile_main")
    adv_amount = st.number_input("Advance amount (₹)", value=0.0, step=0.01, key="adv_amt_main")
    adv_date = st.date_input("Date", value=datetime.date.today(), key="adv_date_main")
    adv_mode = st.selectbox("Mode", ["--Select--","Cash","Card","UPI","Bank Transfer","Cheque"], key="adv_mode_main")
    adv_note = st.text_input("Note (optional)", key="adv_note_main")
    if st.button("Create Advance Note", key="create_adv_btn_main"):
        if not adv_customer_mobile:
            st.error("Customer mobile required")
        elif adv_amount <= 0:
            st.error("Amount must be positive")
        else:
            try:
                if selected_customer == "--Select Customer--" and adv_customer_mobile:
                    save_customer(adv_customer_mobile, adv_customer_mobile, "", "")
                adv_id = create_advance_note(customer_mobile=adv_customer_mobile, amount=adv_amount, date=str(adv_date), mode=(adv_mode if adv_mode != "--Select--" else None), note=adv_note)
                st.success(f"Advance note #{adv_id} created for {adv_customer_mobile}.")
            except Exception as e:
                st.error(f"Could not create advance: {e}")
                st.error(traceback.format_exc())

    st.markdown("---")
    st.subheader("Existing Advances")
    advs = fetch_advances(only_with_remaining=False)

    if advs.empty:
        st.info("No advances")
    else:
        advs = advs.sort_values(["remaining_amount","created_at"], ascending=[False, False]).reset_index(drop=True)
        if "advance_delete_pending" not in st.session_state:
            st.session_state["advance_delete_pending"] = None

        st.write("Advance Notes")
        for _, row in advs.iterrows():
            aid = int(row['id'])
            cols = st.columns([2,1,1,1,1,1])
            cols[0].markdown(f"**#{aid}** — {row['customer_mobile']} — Note: {row.get('note','')}")
            cols[1].markdown(f"Amount: ₹{safe_float(row['amount']):,.2f}")
            cols[2].markdown(f"Remaining: ₹{safe_float(row['remaining_amount']):,.2f}")
            cols[3].markdown(f"Date: {row.get('date')}")
            if cols[4].button("View allocations", key=f"view_alloc_{aid}_main"):
                allocs = fetch_allocations_for_advance(aid)
                if allocs.empty:
                    st.info("No allocations")
                else:
                    st.dataframe(allocs)
            if cols[5].button("Delete Advance", key=f"del_adv_{aid}_main"):
                st.session_state["advance_delete_pending"] = aid

        if st.session_state.get("advance_delete_pending"):
            pending_id = st.session_state["advance_delete_pending"]
            st.markdown("---")
            st.warning(f"You are about to delete advance #{pending_id}. This will remove the advance and its recorded payment.")
            allocs = fetch_allocations_for_advance(pending_id)
            if not allocs.empty:
                st.info("This advance has allocations. Deleting it will also remove those allocations (force delete).")
                st.dataframe(allocs)
            st.checkbox("Force delete (also remove allocations and related payment records)", value=False, key="advance_delete_force_main")
            col_confirm, col_cancel = st.columns([1,1])
            if col_confirm.button("Confirm Delete", key=f"confirm_delete_{pending_id}_main"):
                try:
                    force_flag = bool(st.session_state.get("advance_delete_force_main", False))
                    if not force_flag and not allocs.empty:
                        st.error("Cannot delete: advance has allocations. Check 'Force delete' to allow removal of allocations.")
                    else:
                        delete_advance(pending_id, force=force_flag)
                        st.success(f"Advance #{pending_id} deleted.")
                        if "advance_delete_pending" in st.session_state:
                            del st.session_state["advance_delete_pending"]
                        if "advance_delete_force_main" in st.session_state:
                            del st.session_state["advance_delete_force_main"]
                        advs = fetch_advances(only_with_remaining=False)
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"Could not delete advance: {e}")
                    st.error(traceback.format_exc())
            if col_cancel.button("Cancel", key=f"cancel_delete_{pending_id}_main"):
                if "advance_delete_pending" in st.session_state:
                    del st.session_state["advance_delete_pending"]
                if "advance_delete_force_main" in st.session_state:
                    del st.session_state["advance_delete_force_main"]

        st.markdown("---")
        st.subheader("Allocate Advance to Invoice")
        if not advs.empty:
            adv_choice = st.selectbox("Select advance id", options=advs["id"].tolist(), key="adv_alloc_choice_main")
            if adv_choice:
                adv_row = advs[advs["id"] == adv_choice].iloc[0]
                st.write(f"Advance #{adv_choice} — Customer: {adv_row['customer_mobile']} — Remaining: ₹{safe_float(adv_row['remaining_amount']):,.2f}")
                conn = get_conn()
                try:
                    df_inv = pd.read_sql_query("SELECT invoice_no, date, COALESCE(grand_total,0) AS grand_total, COALESCE(status,'') AS status FROM invoices WHERE customer_mobile=? AND COALESCE(status,'Active') != 'Cancelled' ORDER BY date DESC", conn, params=(adv_row['customer_mobile'],))
                finally:
                    conn.close()
                inv_opts = ["--Select invoice--"] + (df_inv["invoice_no"].tolist() if not df_inv.empty else [])
                inv_sel = st.selectbox("Choose invoice to allocate to", inv_opts, key="adv_alloc_invoice_main")
                default_alloc = float(safe_float(adv_row['remaining_amount']))
                alloc_amt = st.number_input("Allocation amount (₹)", value=default_alloc if default_alloc>0 else 0.0, step=0.01, key="alloc_amt_main2")
                if st.button("Allocate Advance to Invoice", key="alloc_adv_btn_main"):
                    if inv_sel == "--Select invoice--":
                        st.error("Select an invoice")
                    elif alloc_amt <= 0:
                        st.error("Allocation must be positive")
                    else:
                        try:
                            allocate_advance_to_invoice(advance_id=adv_choice, invoice_no=inv_sel, amount=alloc_amt, date=str(datetime.date.today()))
                            st.success(f"Allocated ₹{alloc_amt:,.2f} from advance #{adv_choice} to invoice {inv_sel}.")
                            advs = fetch_advances(only_with_remaining=False)
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass
                        except Exception as e:
                            st.error(f"Could not allocate: {e}")
                            st.error(traceback.format_exc())

# ---------------- Outstanding Summary ----------------
elif page == "Outstanding Summary" or page == "🧾 Outstanding Summary":
    st.title("Outstanding Receivables — Customer Summary")
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
        if not df_summary.empty:
            df_summary = df_summary.sort_values("outstanding", ascending=False).reset_index(drop=True)
    except Exception as e:
        st.error(f"Could not compute outstanding summary: {e}")
        st.error(traceback.format_exc())
        df_summary = pd.DataFrame()
    finally:
        conn.close()

    st.markdown("**Filters**")
    show_only_positive = st.checkbox("Show only customers with outstanding > 0", value=True, key="os_show_positive_main")
    min_outstanding = st.number_input("Minimum outstanding (₹) to include", value=0.0, step=1.0, key="os_min_main")
    if not df_summary.empty:
        df_display = df_summary.copy()
        if show_only_positive: df_display = df_display[df_display["outstanding"] > 0]
        if min_outstanding and min_outstanding > 0: df_display = df_display[df_display["outstanding"] >= float(min_outstanding)]
        st.dataframe(df_display[["customer_mobile","customer_name","total_invoiced","total_paid_applied","outstanding"]])
        total_invoiced_all = df_display["total_invoiced"].sum(); total_paid_all = df_display["total_paid_applied"].sum(); total_outstanding_all = df_display["outstanding"].sum()
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Invoiced (shown)", f"₹{total_invoiced_all:,.2f}")
        c2.metric("Total Paid (applied)", f"₹{total_paid_all:,.2f}")
        c3.metric("Total Outstanding (shown)", f"₹{total_outstanding_all:,.2f}")
        csv = df_display.to_csv(index=False)
        st.download_button("Download summary CSV", data=csv, file_name="outstanding_summary.csv", mime="text/csv", key="os_download")

# ---------------- Create Invoice (with PDF generation + auto-download) ----------------
elif page == "Create Invoice":
    st.header("Create Sales Invoice")

    import base64, json, traceback, datetime
    import streamlit.components.v1 as components
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    # --- Helpers ---
    def safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    def safe_rerun():
        try:
            st.rerun()
        except AttributeError:
            try:
                st.experimental_rerun()
            except Exception:
                pass

    def auto_download_pdf(pdf_bytes, filename="invoice.pdf"):
        """Trigger automatic PDF download in browser via JS"""
        try:
            data_bytes = pdf_bytes.getvalue() if hasattr(pdf_bytes, "getvalue") else bytes(pdf_bytes)
            b64 = base64.b64encode(data_bytes).decode()
            b64_js = json.dumps(b64)
            filename_js = json.dumps(filename)

            html = f"""
            <a id="dl_anchor" style="display:none"></a>
            <script>
            (function(){{
                try {{
                    const b64 = {b64_js};
                    const filename = {filename_js};
                    const href = 'data:application/pdf;base64,' + b64;
                    const a = document.getElementById('dl_anchor') || document.createElement('a');
                    a.id = 'dl_anchor';
                    a.href = href;
                    a.download = filename;
                    a.style.display = 'none';
                    document.body.appendChild(a);
                    a.click();
                    const newWin = window.open(href, '_blank');
                    if (!newWin) {{
                        console.log('Popup blocked');
                    }}
                }} catch (err) {{
                    console.log('Auto-download failed:', err);
                }}
            }})();
            </script>
            """
            components.html(html, height=0)
        except Exception as e:
            st.warning(f"Auto-download failed: {e}")

    # --- Stock Master check ---
    stocks_df = fetch_stocks_df() if 'fetch_stocks_df' in globals() else pd.DataFrame()
    if stocks_df.empty:
        st.error("Cannot create invoices: Stock Master is empty. Please add stock items first.")
        st.stop()

    # --- Start / Clear draft ---
    col1, col2 = st.columns([2, 1])
    with col1:
        if not st.session_state.get("draft_invoice_no"):
            if st.button("Start New Invoice", key="start_new_invoice_btn"):
                try:
                    st.session_state["draft_invoice_no"] = next_invoice_no()
                except Exception:
                    st.session_state["draft_invoice_no"] = f"DRAFT-{int(datetime.datetime.now().timestamp())}"
                st.session_state["invoice_items"] = []
                st.success(f"Draft started: {st.session_state['draft_invoice_no']}")
        else:
            st.info(f"Draft invoice: {st.session_state['draft_invoice_no']}")
            if st.button("Clear Draft", key="clear_draft_btn"):
                st.session_state["draft_invoice_no"] = None
                st.session_state["invoice_items"] = []
    with col2:
        inv_date = st.date_input("Invoice Date", value=datetime.date.today(), key="inv_date_create")

    if not st.session_state.get("draft_invoice_no"):
        st.info("Start a new invoice to add items.")
        st.stop()

    invoice_no = st.session_state["draft_invoice_no"]
    st.subheader(f"Invoice No: {invoice_no}   |   Date: {inv_date.strftime('%d-%m-%Y')}")

    # --- Customer selection ---
    customers_df = fetch_customers_df() if 'fetch_customers_df' in globals() else pd.DataFrame()
    cust_options = ["--New Customer--"] + (customers_df["mobile"].astype(str).tolist() if not customers_df.empty else [])
    selected_customer = st.selectbox("Select customer (mobile)", cust_options, key="create_inv_customer")

    customer_mobile = ""; customer_name = ""; customer_gstin = ""; customer_address = ""; update_customer_master_checkbox = False
    if selected_customer != "--New Customer--" and not customers_df.empty:
        try:
            cust = customers_df[customers_df["mobile"] == selected_customer].iloc[0]
            customer_mobile = cust.get("mobile", ""); customer_name = cust.get("name", "")
            customer_gstin = cust.get("gstin", ""); customer_address = cust.get("address", "")
        except Exception:
            customer_mobile = selected_customer
        st.markdown(f"**Using customer:** {customer_name} — {customer_mobile}")
        with st.expander("View / Edit selected customer (saved only if you tick update)"):
            customer_mobile = st.text_input("Mobile", value=customer_mobile, key="create_inv_selected_mobile")
            customer_name = st.text_input("Name", value=customer_name, key="create_inv_selected_name")
            customer_gstin = st.text_input("GSTIN", value=customer_gstin, key="create_inv_selected_gstin")
            customer_address = st.text_area("Address", value=customer_address, key="create_inv_selected_addr")
            update_customer_master_checkbox = st.checkbox("Update customer master", value=False, key="create_inv_update_cust_master")
    else:
        customer_mobile = st.text_input("Mobile", key="create_inv_mobile")
        customer_name = st.text_input("Name", key="create_inv_name")
        customer_gstin = st.text_input("GSTIN", key="create_inv_gstin")
        customer_address = st.text_area("Address", key="create_inv_addr")

    st.markdown("---")
    st.subheader("Add Item")

    # --- Add Item (purity restricted) ---
    left, right = st.columns([1.4, 2])
    with left:
        cat = st.selectbox("Category", ["Gold Ornaments", "Silver Ornaments", "Diamond Ornaments"], key="new_item_cat")
        unit = "Grms" if cat in ["Gold Ornaments", "Silver Ornaments"] else "Ct"

        try:
            available_purities = (
                stocks_df[stocks_df["category"] == cat]["purity"]
                .dropna().astype(str).str.strip().unique().tolist()
            )
        except Exception:
            available_purities = []

        if not available_purities:
            st.error(f"No stock exists in Stock Master for category '{cat}'. Please add stock first.")
            purity, matching = None, pd.DataFrame()
        else:
            purity = st.selectbox("Purity", available_purities, key="new_item_purity")
            try:
                matching = stocks_df[(stocks_df["category"] == cat) & (stocks_df["purity"] == purity)].copy()
            except Exception:
                matching = pd.DataFrame()

        stock_id, avail = None, None
        if purity is not None and not matching.empty:
            stock_opts = ["--Select stock--"] + matching.apply(
                lambda r: f"{r['id']} | {r.get('description','')} | Avail:{r.get('quantity',0)} {r.get('unit','')}", axis=1
            ).tolist()
            stock_sel = st.selectbox("Choose stock (required)", stock_opts, key="new_item_stocksel")
            if stock_sel != "--Select stock--":
                try:
                    stock_id = int(stock_sel.split("|")[0].strip())
                    row = matching[matching["id"] == stock_id]
                    avail = float(row.iloc[0].get("quantity", 0.0) or 0.0) if not row.empty else None
                except Exception:
                    stock_id, avail = None, None

        description = st.text_input("Description", "", key="create_item_description")
        qty = st.number_input(f"Qty ({unit})", 0.0, step=0.01, key="new_item_qty")
        rate = st.number_input("Rate (₹)", 0.0, step=0.01, key="new_item_rate")
        making = st.number_input("Making (₹)", 0.0, step=0.01, key="new_item_making")
        if (avail is not None) and qty > avail:
            st.warning(f"Qty exceeds available stock ({avail})")

        can_add_item = (purity is not None) and (stock_id is not None)
        if not can_add_item:
            st.caption("You must select a purity and stock entry to add an item.")
        else:
            if st.button("Add item to invoice", key="add_item_btn"):
                if not description.strip():
                    st.error("Description is mandatory.")
                elif qty <= 0:
                    st.error("Enter valid quantity.")
                elif rate < 0:
                    st.error("Enter valid rate.")
                elif (avail is not None) and qty > avail:
                    st.error(f"Cannot add: qty {qty} > stock {avail}.")
                else:
                    line_amount = safe_float(qty) * safe_float(rate) + safe_float(making)
                    st.session_state.setdefault("invoice_items", [])
                    st.session_state["invoice_items"].append({
                        "stock_id": stock_id, "category": cat, "purity": purity,
                        "item_name": description.strip(), "qty": float(qty),
                        "unit": unit, "rate": float(rate), "making": float(making),
                        "amount": float(line_amount)
                    })
                    st.success("Item added.")

    with right:
        st.subheader("Invoice Items")
        items = st.session_state.get("invoice_items", [])
        if not items:
            st.info("No items yet.")
        else:
            df_items = pd.DataFrame(items)
            st.dataframe(df_items[["stock_id","category","purity","item_name","qty","unit","rate","making","amount"]])
            remove_index = st.number_input("Remove item index (1-based)", 0, len(items), 0, 1, key="remove_idx")
            if st.button("Remove selected item", key="remove_item_btn") and remove_index > 0:
                st.session_state["invoice_items"].pop(remove_index-1)
                st.success("Removed item.")

    st.markdown("---")
    st.subheader("Totals & Payment")
    gst_rate = st.number_input("GST %", 3.0, step=0.1, key="gst_rate_create")
    gst_type = st.selectbox("GST Type", ["Intra-State","Inter-State"], key="gst_type_create")
    subtotal = sum([safe_float(x.get("amount",0)) for x in st.session_state.get("invoice_items",[])])
    gst_total = subtotal * (gst_rate/100.0)
    grand_total = subtotal + gst_total
    st.write(f"Subtotal: ₹{subtotal:,.2f} | GST: ₹{gst_total:,.2f} | Grand: ₹{grand_total:,.2f}")

    pay_received = st.number_input("Payment received now (₹)", 0.0, step=0.01, key="pay_received_create")
    pay_mode = st.selectbox("Payment mode", ["--Select--","Cash","Card","UPI","Bank Transfer","Cheque"], key="pay_mode_create")
    pay_note = st.text_input("Payment note (optional)", key="pay_note_create")

    if st.button("Save Invoice", key="save_invoice_btn"):
        if not st.session_state.get("invoice_items"):
            st.error("Add at least one item before saving.")
        elif not customer_mobile.strip():
            st.error("Customer mobile is required.")
        else:
            try:
                # Update customer if requested
                if update_customer_master_checkbox:
                    save_customer(customer_mobile, customer_name, customer_gstin, customer_address)

                inv_no = insert_invoice_and_items(
                    invoice_no_requested=invoice_no, date_str=str(inv_date),
                    customer_mobile=customer_mobile, customer_name=customer_name,
                    customer_gstin=customer_gstin, customer_address=customer_address,
                    gst_rate=gst_rate, gst_type=gst_type,
                    items=st.session_state["invoice_items"],
                    payment_status="Paid" if pay_received>0 else "Unpaid",
                    payment_mode=(pay_mode if pay_mode!="--Select--" else None),
                    payment_received=pay_received,
                    payment_date=str(datetime.date.today()),
                    payment_note=pay_note
                )

                # Ensure customer saved in master
                try:
                    save_customer(customer_mobile, customer_name, customer_gstin, customer_address)
                except Exception:
                    pass

                st.success(f"Invoice {inv_no} saved.")

                # Generate PDF
                pdf_bytes = None
                try:
                    comp = fetch_company() if 'fetch_company' in globals() else None
                    inv_row, items_rows = read_invoice_from_db(inv_no)
                    pdf_bytes = generate_invoice_pdf(inv_no, inv_row, items_rows, company_row=comp)
                except Exception as e_pdf:
                    st.warning(f"PDF generation failed: {e_pdf}")
                    st.error(traceback.format_exc())

                if pdf_bytes:
                    # Auto-download
                    auto_download_pdf(pdf_bytes, f"{inv_no}.pdf")

                    # Fallback download button
                    try:
                        data_bytes = pdf_bytes.getvalue() if hasattr(pdf_bytes, "getvalue") else bytes(pdf_bytes)
                        st.download_button("Download Invoice PDF (if auto-download blocked)", data=data_bytes, file_name=f"{inv_no}.pdf", mime="application/pdf")
                    except Exception:
                        st.warning("Could not create fallback download button.")

                              # ---------------- Auto-start NEW invoice (immediately begin a fresh invoice) ----------------
                try:
                    # Attempt to get the next sequential invoice number from counter
                    new_invoice_no = next_invoice_no()
                except Exception:
                    # Fallback: timestamped label (shouldn't normally happen)
                    new_invoice_no = f"DRAFT-{int(datetime.datetime.now().timestamp())}"

                # Record the invoice we just saved
                st.session_state["last_saved_invoice"] = inv_no

                # Make the new invoice the active invoice (reuse your draft key so UI behaves the same)
                st.session_state["draft_invoice_no"] = new_invoice_no

                # Clear any existing items so the form is fresh for the new invoice
                st.session_state["invoice_items"] = []

                # Set invoice date to today for the new invoice
                try:
                    st.session_state["inv_date_create"] = datetime.date.today()
                except Exception:
                    # If your date key differs or can't be set, ignore silently
                    pass

                # Inform user that a new invoice has started (live invoice)
                st.success(f"Started new invoice: {st.session_state['draft_invoice_no']}")

                # Ensure the app remains on Create Invoice page (if you use 'page' session key)
                try:
                    st.session_state["page"] = "Create Invoice"
                except Exception:
                    pass

                # Trigger a rerun so the Create Invoice UI resets for the new invoice
                try:
                    safe_rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        # If rerun fails, just continue — UI will still show the new invoice number
                        pass
                # ---------------- end auto-start new invoice ----------------

                
            except Exception as e:
                st.error(f"Could not save invoice: {e}")
                st.error(traceback.format_exc())

# ---------------- Sales Register (rest of pages unchanged, using tenant-awae DB) ----------------
elif page == "Sales Register" or page == "🛒 Sales Register":
    st.header("Sales Register — Invoices & Itemwise CSV Export")
    conn = get_conn()
    try:
        df_all = pd.read_sql_query("SELECT invoice_no,date,customer_name,customer_mobile,customer_gstin,COALESCE(grand_total,0) AS grand_total, COALESCE(status,'') as status FROM invoices ORDER BY date DESC LIMIT 5000", conn)
    finally:
        conn.close()

    if df_all.empty:
        st.info("No invoices yet.")
    else:
        col_f1, col_f2, col_f3 = st.columns([3,2,2])
        q = col_f1.text_input("Search invoice no / customer mobile / name", key="sales_search_q")
        filt_status = col_f2.selectbox("Status", ["All","Active","Cancelled"], index=0, key="sales_filter_status")
        min_amt = col_f3.number_input("Min amount", value=0.0, step=1.0, key="sales_min_amt")

        st.markdown("**Date range for export / filtering**")
        dr_col1, dr_col2, dr_col3, dr_col4 = st.columns([1,1,1,2])
        if dr_col1.button("Today"):
            start_date = end_date = datetime.date.today()
            st.session_state["_sales_range_quick"] = "today"
        if dr_col2.button("This month"):
            today = datetime.date.today()
            start_date = today.replace(day=1)
            end_date = today
            st.session_state["_sales_range_quick"] = "this_month"
        if dr_col3.button("Previous month"):
            today = datetime.date.today()
            first_of_current = today.replace(day=1)
            prev_last = first_of_current - datetime.timedelta(days=1)
            start_date = prev_last.replace(day=1)
            end_date = prev_last
            st.session_state["_sales_range_quick"] = "prev_month"
        dr_sel = dr_col4.selectbox("Or choose range type", ["Use quick preset above","Custom range"], key="sales_dr_sel")
        if dr_sel == "Custom range":
            cd = st.columns(2)
            start_date = cd[0].date_input("Start date", value=(st.session_state.get("_sales_start") or (datetime.date.today() - datetime.timedelta(days=30))), key="sales_custom_start")
            end_date = cd[1].date_input("End date", value=(st.session_state.get("_sales_end") or datetime.date.today()), key="sales_custom_end")
            st.session_state["_sales_start"] = start_date; st.session_state["_sales_end"] = end_date
        else:
            if "_sales_range_quick" not in st.session_state:
                start_date = datetime.date.today() - datetime.timedelta(days=30); end_date = datetime.date.today()
            else:
                if st.session_state.get("_sales_range_quick") == "today":
                    start_date = end_date = datetime.date.today()
                elif st.session_state.get("_sales_range_quick") == "this_month":
                    t = datetime.date.today(); start_date = t.replace(day=1); end_date = t
                elif st.session_state.get("_sales_range_quick") == "prev_month":
                    t = datetime.date.today(); first_of_current = t.replace(day=1); prev_last = first_of_current - datetime.timedelta(days=1); start_date = prev_last.replace(day=1); end_date = prev_last
                else:
                    start_date = datetime.date.today() - datetime.timedelta(days=30); end_date = datetime.date.today()

        st.markdown(f"**Filtering invoices from {start_date} to {end_date}**")

        df = df_all.copy()
        if q:
            qlow = q.lower()
            df = df[df.apply(lambda r: qlow in str(r['invoice_no']).lower() or qlow in str(r['customer_mobile']).lower() or qlow in str(r['customer_name']).lower(), axis=1)]
        if filt_status != "All":
            df = df[df['status']==filt_status]
        if min_amt and min_amt > 0:
            df = df[df['grand_total'] >= float(min_amt)]

        try:
            df['date'] = pd.to_datetime(df['date']).dt.date
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        except Exception:
            st.warning("Could not parse invoice dates for filtering; CSV will include all visible rows.")

        df = df.sort_values("date", ascending=False)
        st.dataframe(df)

        st.markdown("### Export itemwise CSV for the selected date range")
        if st.button("Generate item-wise CSV (all invoices in date range)"):
            try:
                conn = get_conn()
                sql = """
                    SELECT inv.invoice_no, inv.date, inv.customer_name, inv.customer_mobile, inv.customer_gstin,
                           inv.gst_rate, inv.gst_type, inv.subtotal, inv.grand_total, ii.category, ii.purity, ii.hsn,
                           ii.item_name, ii.qty, ii.unit, ii.rate, ii.making, ii.amount
                    FROM invoices inv
                    LEFT JOIN invoice_items ii ON ii.invoice_no = inv.invoice_no
                    WHERE date(inv.date) BETWEEN ? AND ?
                    ORDER BY inv.date DESC, inv.invoice_no, ii.id
                """
                df_items = pd.read_sql_query(sql, conn, params=(str(start_date), str(end_date)))
                conn.close()
                if df_items.empty:
                    st.info("No invoice items found in this date range.")
                else:
                    if 'customer_gstin' not in df_items.columns:
                        df_items['customer_gstin'] = ""
                    df_items['customer_gstin'] = df_items['customer_gstin'].fillna("")
                    csv_bytes = df_items.to_csv(index=False).encode('utf-8')
                    filename = f"sales_itemwise_{start_date}_{end_date}.csv"
                    st.download_button("Download item-wise CSV", data=csv_bytes, file_name=filename, mime="text/csv")
                    st.success(f"Prepared {len(df_items)} item lines across invoices.")
            except Exception as e:
                st.error(f"Could not generate CSV: {e}")
                st.error(traceback.format_exc())

        sel = st.selectbox("Select invoice for actions", options=["--Select--"] + df["invoice_no"].tolist(), key="sales_sel_invoice")
        if sel and sel != "--Select--":
            inv_row, items_rows = read_invoice_from_db(sel)
            if inv_row is None:
                st.error("Invoice not found.")
            else:
                st.subheader(f"Invoice {sel}")
                st.table(pd.DataFrame(items_rows)[["category","purity","hsn","item_name","qty","unit","rate","making","amount"]])

                conn2 = get_conn()
                try:
                    payments_df = pd.read_sql_query("SELECT * FROM payments WHERE invoice_no=? ORDER BY created_at DESC", conn2, params=(sel,))
                finally:
                    conn2.close()
                if not payments_df.empty:
                    st.subheader("Payments for this invoice")
                    st.dataframe(payments_df)
                allocs = fetch_allocations_for_invoice(sel)
                if not allocs.empty:
                    st.subheader("Advance allocations to this invoice")
                    st.dataframe(allocs)

                action_col1, action_col2, action_col3 = st.columns(3)
                if action_col1.button("Download PDF", key=f"download_pdf_{sel}"):
                    comp = fetch_company()
                    pdf_bytes = generate_invoice_pdf(sel, inv_row, items_rows, company_row=comp)
                    st.download_button("Download Invoice PDF", data=pdf_bytes, file_name=f"{sel}.pdf", mime="application/pdf", key=f"download_file_{sel}")
                if action_col2.button("Cancel Invoice", key=f"cancel_inv_{sel}"):
                    try:
                        cancel_invoice(sel)
                        st.success(f"Invoice {sel} cancelled and stock restored.")
                    except Exception as e:
                        st.error(f"Could not cancel: {e}")
                        st.error(traceback.format_exc())

                toggle_key = f"show_pay_panel_{sel}"
                if toggle_key not in st.session_state:
                    st.session_state[toggle_key] = False
                if action_col3.button("Mark Payment / Apply Payment", key=f"mark_pay_{sel}"):
                    st.session_state[toggle_key] = not st.session_state[toggle_key]

                if st.session_state.get(toggle_key):
                    st.markdown("---")
                    st.subheader("Apply Payment — Record transaction against invoice")
                    conn3 = get_conn()
                    try:
                        paid_df = pd.read_sql_query("SELECT COALESCE(SUM(amount),0) as paid FROM payments WHERE invoice_no=? AND COALESCE(is_advance,0)=0", conn3, params=(sel,))
                        already_paid = float(paid_df['paid'].iloc[0]) if not paid_df.empty else 0.0
                    finally:
                        conn3.close()
                    remaining = float(safe_float(inv_row.get('grand_total',0.0)) - already_paid)
                    amt_key = f"pay_amount_{sel}"
                    mode_key = f"pay_mode_{sel}"
                    note_key = f"pay_note_{sel}"
                    amount = st.number_input("Payment amount (₹)", value=remaining if remaining>0 else 0.0, step=0.01, key=amt_key)
                    mode = st.selectbox("Mode", ["Cash","Card","UPI","Bank Transfer","Cheque"], key=mode_key)
                    note = st.text_input("Note (optional)", key=note_key)
                    if st.button("Apply Payment Now", key=f"apply_now_{sel}"):
                        try:
                            add_payment(invoice_no=sel, amount=amount, date=str(datetime.date.today()), mode=mode, note=note or "Manual payment applied", is_advance=False, customer_mobile=inv_row.get("customer_mobile"))
                            conn4 = get_conn(); cur = conn4.cursor()
                            cur.execute("SELECT COALESCE(SUM(amount),0) FROM payments WHERE invoice_no=? AND COALESCE(is_advance,0)=0", (sel,))
                            applied_sum = float(cur.fetchone()[0] or 0.0)
                            grand_total = float(inv_row.get('grand_total',0.0) or 0.0)
                            payment_status = 'Paid' if applied_sum + 1e-9 >= grand_total and grand_total>0 else ('Partially Paid' if applied_sum>0 else 'Unpaid')
                            cur.execute("UPDATE invoices SET payment_received=?, payment_status=?, payment_mode=? WHERE invoice_no=?", (applied_sum, payment_status, mode, sel))
                            conn4.commit(); conn4.close()
                            st.success("Payment recorded and invoice updated.")
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass
                        except Exception as e:
                            st.error(f"Could not record payment: {e}")

# ---------------- Stock Master ----------------
elif page == "Stock Master" or page == "📦 Stock Master":
    st.header("Inventory Management — Stock Master")
    stocks_df = fetch_stocks_df()
    with st.expander("Add / Update Stock (datewise)"):
        c1, c2 = st.columns(2)
        with c1:
            category = st.selectbox("Category", ["Gold Ornaments","Silver Ornaments","Diamond Ornaments"], key="addstock_category_main")
            if category == "Gold Ornaments":
                purity = st.selectbox("Purity", ["22K","20K","18K","14K","10K"], key="addstock_purity_main")
                unit = "Grms"
            elif category == "Silver Ornaments":
                purity = "Std"; unit = "Grms"
            else:
                purity = "Std"; unit = "Ct"
            description = st.text_input("Description", key="addstock_desc_main")
        with c2:
            tx_date = st.date_input("Transaction Date", value=datetime.date.today(), key="addstock_txdate_main")
            qty = st.number_input("Quantity (positive add, negative reduce)", value=0.0, step=0.01, key="addstock_qty_main")
            reason = st.text_input("Reason", value="Opening / Adjustment", key="addstock_reason_main")
        if st.button("Add / Update Stock", key="addstock_btn_main"):
            if not description:
                st.error("Description required")
            else:
                sid, newqty = add_or_update_stock(category, purity, description, unit, qty, tx_date=str(tx_date), reason=reason)
                st.success(f"Stock updated (id {sid}) -> New qty: {newqty}")
                stocks_df = fetch_stocks_df()

    st.subheader("Current Stock Overview")
    st.dataframe(stocks_df)

    st.subheader("Edit Stock Master (select a row to update)")
    if not stocks_df.empty:
        options = stocks_df.apply(lambda r: f"{r['id']}|{r['category']}|{r['purity']}|{r['description']}|Qty:{r['quantity']}", axis=1).tolist()
        sel = st.selectbox("Select stock to edit", options, key="select_stock_to_edit_main")
        if sel:
            sid = int(sel.split("|")[0])
            conn = get_conn(); cur = conn.cursor()
            cur.execute("SELECT * FROM stocks WHERE id=?", (sid,))
            srow = cur.fetchone(); conn.close()
            if srow:
                st.markdown(f"**Editing stock id {sid}**")
                cat_key = f"edit_category_{sid}_main"
                purity_key = f"edit_purity_{sid}_main"
                desc_key = f"edit_description_{sid}_main"
                unit_key = f"edit_unit_{sid}_main"
                qty_mode_key = f"edit_qty_mode_{sid}_main"
                new_qty_key = f"edit_new_qty_{sid}_main"
                delta_key = f"edit_delta_{sid}_main"
                save_btn_key = f"edit_save_{sid}_main"
                categories = ["Gold Ornaments","Silver Ornaments","Diamond Ornaments"]
                try:
                    default_cat_idx = categories.index(srow[1]) if srow[1] in categories else 0
                except Exception:
                    default_cat_idx = 0
                new_category = st.selectbox("Category", categories, index=default_cat_idx, key=cat_key)
                new_purity = st.text_input("Purity", value=srow[2] or "", key=purity_key)
                new_description = st.text_input("Description", value=srow[3] or "", key=desc_key)
                new_unit = st.text_input("Unit", value=srow[4] or "", key=unit_key)
                qty_mode = st.radio("Quantity change", ["Set absolute quantity", "Apply delta (add/subtract)"], key=qty_mode_key)
                current_qty = float(srow[5] or 0.0)
                if qty_mode == "Set absolute quantity":
                    new_qty = st.number_input("New absolute quantity", value=current_qty, key=new_qty_key)
                    qty_delta = float(new_qty) - current_qty
                else:
                    qty_delta = st.number_input("Delta to apply (positive add, negative reduce)", value=0.0, key=delta_key)
                    new_qty = current_qty + float(qty_delta)
                if st.button("Save stock changes", key=save_btn_key):
                    try:
                        conn = get_conn(); cur = conn.cursor()
                        cur.execute("UPDATE stocks SET category=?, purity=?, description=?, unit=?, quantity=?, created_at=? WHERE id=?", (new_category, new_purity, new_description, new_unit, float(new_qty), str(datetime.datetime.now()), sid))
                        cur.execute("PRAGMA table_info(stock_transactions)")
                        tx_cols = [r[1] for r in cur.fetchall()]
                        tx_data = {"stock_id":sid,"tx_date":str(datetime.date.today()),"change":float(qty_delta),"reason":"Manual edit","resulting_qty":float(new_qty)}
                        if "created_at" in tx_cols:
                            tx_data["created_at"] = str(datetime.datetime.now())
                        insert_dict_dynamic(conn, "stock_transactions", tx_data)
                        conn.commit(); conn.close()
                        st.success("Stock updated")
                        stocks_df = fetch_stocks_df()
                    except Exception as e:
                        st.error(f"Could not update stock: {e}")

        st.markdown("---")
        st.subheader("All Stock Movements (All items)")
        sm_col1, sm_col2, sm_col3 = st.columns([1,1,2])
        if sm_col1.button("Today", key="sm_today"):
            sm_start = sm_end = datetime.date.today()
            st.session_state["_sm_quick"] = "today"
        if sm_col2.button("This month", key="sm_this_month"):
            t = datetime.date.today(); sm_start = t.replace(day=1); sm_end = t; st.session_state["_sm_quick"] = "this_month"
        sm_choice = sm_col3.selectbox("Filter type", ["Last 30 days","Custom range"], key="sm_choice")
        if sm_choice == "Custom range":
            scol, ecol = st.columns(2)
            sm_start = scol.date_input("Start", value=(st.session_state.get("_sm_start") or (datetime.date.today() - datetime.timedelta(days=30))), key="sm_custom_start")
            sm_end = ecol.date_input("End", value=(st.session_state.get("_sm_end") or datetime.date.today()), key="sm_custom_end")
            st.session_state["_sm_start"] = sm_start; st.session_state["_sm_end"] = sm_end
        else:
            if "_sm_quick" not in st.session_state:
                sm_start = datetime.date.today() - datetime.timedelta(days=30); sm_end = datetime.date.today()
            else:
                if st.session_state.get("_sm_quick") == "today":
                    sm_start = sm_end = datetime.date.today()
                elif st.session_state.get("_sm_quick") == "this_month":
                    t = datetime.date.today(); sm_start = t.replace(day=1); sm_end = t
                else:
                    sm_start = datetime.date.today() - datetime.timedelta(days=30); sm_end = datetime.date.today()

        st.markdown(f"Showing stock transactions from {sm_start} to {sm_end}")

        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(stock_transactions)")
            cols_info = cur.fetchall()
            tx_cols = [r[1] for r in cols_info]
            pick_cols = [c for c in ["id","stock_id","tx_date","change","reason","resulting_qty","created_at"] if c in tx_cols]
            if not pick_cols:
                st.info("No stock_transactions columns available.")
            else:
                select_clause = ", ".join(pick_cols)
                q = f"SELECT {select_clause} FROM stock_transactions ORDER BY tx_date DESC, id DESC"
                tx_df = pd.read_sql_query(q, conn)
                if "tx_date" in tx_df.columns:
                    try:
                        tx_df['tx_date'] = pd.to_datetime(tx_df['tx_date']).dt.date
                        tx_df = tx_df[(tx_df['tx_date'] >= sm_start) & (tx_df['tx_date'] <= sm_end)]
                    except Exception:
                        pass
                if "stock_id" in tx_df.columns:
                    stocks_ref = fetch_stocks_df()
                    if not stocks_ref.empty:
                        stocks_ref = stocks_ref.rename(columns={"id":"stock_id","description":"stock_description","category":"stock_category","purity":"stock_purity","unit":"stock_unit"})
                        tx_df = tx_df.merge(stocks_ref[["stock_id","stock_description","stock_category","stock_purity","stock_unit"]], on="stock_id", how="left")
                if tx_df.empty:
                    st.info("No stock transactions for the selected range.")
                else:
                    st.dataframe(tx_df)
                    csv_bytes = tx_df.to_csv(index=False).encode('utf-8')
                    fn = f"stock_movements_{sm_start}_{sm_end}.csv"
                    st.download_button("Download stock movements CSV", data=csv_bytes, file_name=fn, mime="text/csv")
            conn.close()
        except Exception as e:
            st.error(f"Could not fetch transactions: {e}")
            st.error(traceback.format_exc())
    else:
        st.info("No stock records to edit.")

# ---------------- Customer Master ----------------
elif page == "Customer Master":
    st.header("Customer Directory")
    with st.expander("Add / Update Customer"):
        mob = st.text_input("Mobile", key="cust_mobile_main")
        nm = st.text_input("Name", key="cust_name_main")
        gst = st.text_input("GSTIN", key="cust_gstin_main")
        address = st.text_area("Address", key="cust_address_main")
        if st.button("Save Customer", key="save_customer_btn_main"):
            if not mob or not nm:
                st.error("Mobile and Name required")
            else:
                save_customer(mob, nm, gst, address)
                st.success("Customer saved")
    st.subheader("Registered Customers")
    st.dataframe(fetch_customers_df())

# ---------------- Invoice History ----------------
elif page == "Invoice History":
    st.header("Invoice Archive — Historical Records")
    conn = get_conn()
    try:
        df_all = pd.read_sql_query("SELECT invoice_no,date,customer_name,customer_mobile,COALESCE(grand_total,0) AS grand_total,status FROM invoices ORDER BY date DESC LIMIT 500", conn)
    finally:
        conn.close()
    if df_all.empty:
        st.info("No invoices yet")
    else:
        sel = st.selectbox("Select invoice", ["--Select--"] + df_all["invoice_no"].tolist(), key="inv_hist_sel")
        if sel and sel != "--Select--":
            inv_row, items_rows = read_invoice_from_db(sel)
            if inv_row is None:
                st.error("Invoice not found")
            else:
                st.subheader(f"Invoice {sel}")
                st.table(pd.DataFrame(items_rows)[["category","purity","hsn","item_name","qty","unit","rate","making","amount"]])
                connp = get_conn()
                try:
                    payments_df = pd.read_sql_query("SELECT * FROM payments WHERE invoice_no=?", connp, params=(sel,))
                finally:
                    connp.close()
                if not payments_df.empty:
                    st.subheader("Payments for this invoice")
                    st.dataframe(payments_df)
                allocs = fetch_allocations_for_invoice(sel)
                if not allocs.empty:
                    st.subheader("Advance allocations to this invoice")
                    st.dataframe(allocs)

# ---------------- Payments Ledger ----------------
elif page == "Payments Ledger" or page == "💳 Payments Ledger":
    st.header("Payments Ledger — Transactions & Reconciliation")
    conn = get_conn()
    try:
        df_pay = pd.read_sql_query("SELECT * FROM payments ORDER BY created_at DESC LIMIT 1000", conn)
    finally:
        conn.close()
    if df_pay.empty:
        st.info("No payments recorded yet.")
    else:
        st.dataframe(df_pay)
        pid = st.number_input("Enter payment id to edit (or 0 to skip)", min_value=0, value=0, step=1, key="payments_edit_pid")
        if pid and pid > 0:
            conn = get_conn(); cur = conn.cursor()
            cur.execute("SELECT * FROM payments WHERE id=?", (pid,))
            prow = cur.fetchone(); conn.close()
            if not prow:
                st.error("Payment id not found.")
            else:
                st.markdown(f"Editing payment id {pid}")
                invoice_no = st.text_input("Invoice no (blank if advance)", value=prow[1] or "", key=f"pay_invoice_{pid}")
                customer_mobile = st.text_input("Customer mobile", value=prow[2] or "", key=f"pay_cust_{pid}")
                amount = st.number_input("Amount (₹)", value=float(prow[3] or 0.0), step=0.01, key=f"pay_amount_{pid}_edit")
                pdate = st.date_input("Date", value=pd.to_datetime(prow[4]).date() if prow[4] else datetime.date.today(), key=f"pay_date_{pid}")
                mode = st.selectbox("Mode", ["--Select--","Cash","Card","UPI","Bank Transfer","Cheque","Other"], index=0, key=f"pay_mode_{pid}")
                note = st.text_input("Note", value=prow[6] or "", key=f"pay_note_{pid}")
                is_adv = True if prow[7] == 1 else False
                if st.button("Save payment changes", key=f"save_payment_{pid}"):
                    try:
                        conn = get_conn(); cur = conn.cursor()
                        cur.execute("UPDATE payments SET invoice_no=?, customer_mobile=?, amount=?, date=?, mode=?, note=?, is_advance=? WHERE id=?", (invoice_no or None, customer_mobile or None, float(amount), str(pdate), (mode if mode != "--Select--" else None), note, 1 if is_adv else 0, pid))
                        conn.commit(); conn.close()
                        st.success("Payment updated.")
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass
                    except Exception as e:
                        st.error(f"Could not update payment: {e}")

# ---------------- Company Settings ----------------
elif page == "Company Settings" or page == "🏢 Company Settings":
    st.header("Company Profile & Settings")
    company = fetch_company()
    name = st.text_input("Company Name", value=company[0] if company else "", key="company_name")
    gstin = st.text_input("Company GSTIN", value=company[1] if company else "", key="company_gstin")
    addr = st.text_area("Address", value=company[2] if company else "", key="company_addr")
    logo_file = st.file_uploader("Logo (png/jpg)", type=["png","jpg","jpeg"], key="company_logo")
    sig_file = st.file_uploader("Signature (png/jpg)", type=["png","jpg","jpeg"], key="company_sig")
    if st.button("Save Company", key="save_company_btn"):
        logo_bytes = logo_file.read() if logo_file else (company[3] if company else None)
        sig_bytes = sig_file.read() if sig_file else (company[4] if company else None)
        save_company(name, gstin, addr, logo_bytes, sig_bytes)
        st.success("Company saved")

else:
    st.info("Select a menu item from the sidebar to get started.")

st.markdown("---")
st.caption("GoldTrader Pro - Invoicing & Inventory — simple, secure, professional")
