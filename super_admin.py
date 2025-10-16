# super_admin_improved.py
# Improved Super Admin dashboard for GoldTrader Pro tenancy system
# Left-image (from ./assets) + right-form login layout, and fixed tenant enable/disable flow.

import streamlit as st
import os
import json
import datetime
import pandas as pd
from pathlib import Path

# try import tenants_auth if available
try:
    import tenants_auth
except Exception:
    tenants_auth = None

# ---------- Helpers & Config ----------
BASE_DIR = Path(__file__).resolve().parent
OVERRIDES_FILE = BASE_DIR / "super_admin_overrides.json"
AUDIT_LOG = BASE_DIR / "super_admin_audit.log"

BRAND_NAME = os.environ.get("BRAND_NAME", "GoldTrader Pro")
PREFERRED_NAMES = ["login_bg.png", "login_bg.jpg", "login_bg.jpeg", "login_bg.webp"]
ASSETS_DIR = BASE_DIR / "assets"

# page icon
st.set_page_config(page_title="Super Admin — GoldTrader Pro", layout="wide", page_icon="💎")

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def log_action(super_user, action, details=""):
    ts = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
    entry = f"{ts} | {super_user} | {action} | {details}\n"
    try:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass

def load_overrides():
    if not OVERRIDES_FILE.exists():
        return {}
    try:
        with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_overrides(d):
    try:
        with open(OVERRIDES_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass

def check_super_creds(username, password):
    env_user = os.environ.get("SUPER_ADMIN_USERNAME")
    env_pwd = os.environ.get("SUPER_ADMIN_PASSWORD")
    if not env_user:
        env_user = "GSF"
    if not env_pwd:
        env_pwd = "@Gsf025@"
    return username == env_user and password == env_pwd

def get_tenant_db_path(tenant_id):
    if tenants_auth and hasattr(tenants_auth, "get_tenant_db_path"):
        try:
            return tenants_auth.get_tenant_db_path(tenant_id)
        except Exception:
            pass
    if tenants_auth and hasattr(tenants_auth, "TENANTS_DIR"):
        return os.path.abspath(os.path.join(tenants_auth.TENANTS_DIR, f"invoices_{tenant_id}.db"))
    return os.path.abspath(os.path.join(str(BASE_DIR), f"invoices_{tenant_id}.db"))

# Robust asset finder
def find_asset_image():
    # 1) check preferred names
    for name in PREFERRED_NAMES:
        p = ASSETS_DIR / name
        if p.exists() and p.is_file():
            return str(p)
    # 2) fallback - pick first image file
    if ASSETS_DIR.exists() and ASSETS_DIR.is_dir():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.svg"):
            matches = sorted(ASSETS_DIR.glob(ext))
            if matches:
                return str(matches[0])
    return None

# ---------- Super Admin Login (left image + right form) ----------
if "super_auth" not in st.session_state:
    st.session_state.super_auth = False
    st.session_state.super_user = None

if not st.session_state.super_auth:
    st.set_page_config(page_title=f"{BRAND_NAME} — Super Admin Sign in", layout="wide", page_icon="💎")

    # CSS (keeps styling consistent)
    st.markdown(
        """
        <style>
        .header-card {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        .logo-circle {
            background: #d4af37; /* gold */
            color: black;
            font-weight: 700;
            font-size: 24px;
            width: 56px;
            height: 56px;
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
            font-size: 16px;
            color: #6b7280;
        }
        .small-note { font-size: 12px; color: #9ca3af; }
        .login-box { padding: 8px 0 4px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([0.8, 1])

    # LEFT: image from assets (robust)
    with left_col:
        img_path = find_asset_image()
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.image("https://via.placeholder.com/600x800.png?text=GoldTrader+Pro+Login", use_container_width=True)

    # RIGHT: login form — using your provided layout snippet
    with right_col:
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='header-card'>"
            "<div class='logo-circle'>GT</div>"
            "<div class='title-block'>"
            f"<div class='title-main'>{BRAND_NAME}</div>"
            "<div class='title-sub'>Admin Console</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True
        )

        with st.form(key="super_login_form"):
            username = st.text_input("Username", key="super_login_usr")
            password = st.text_input("Password", type="password", key="super_login_pwd")
            submitted = st.form_submit_button("Login")

        if submitted:
            if check_super_creds(username, password):
                st.session_state.super_auth = True
                st.session_state.super_user = username
                st.success("Logged in as Super Admin.")
                log_action(username, "LOGIN", "Super admin logged in")
                safe_rerun()
            else:
                st.error("Invalid super admin credentials")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='small-note'>Trouble signing in? Contact the system administrator.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ---------- After login: Main Super Admin App ----------
super_user = st.session_state.super_user
sidebar = st.sidebar
sidebar.markdown(f"**Signed in as:** `{super_user}`")
if sidebar.button("Logout"):
    log_action(super_user, "LOGOUT", "")
    st.session_state.super_auth = False
    st.session_state.super_user = None
    safe_rerun()

# Load overrides once
overrides = load_overrides()

# Top navigation
tab = st.tabs(["Tenants", "Create Tenant", "Users & Passwords", "Overrides", "Audit Log"])

# ---------- TENANTS ----------
with tab[0]:
    st.header("Tenants")
    try:
        tenants = tenants_auth.list_tenants()
    except Exception as e:
        st.error(f"Could not fetch tenants: {e}")
        tenants = []

    if not tenants:
        st.info("No tenants found.")
    else:
        c1, c2 = st.columns([3,1])
        with c1:
            q = st.text_input("Search tenants (name or id)")
        with c2:
            show_disabled = st.checkbox("Show disabled", value=True)

        df = pd.DataFrame(tenants, columns=["id","name","created_at"])
        df["enabled"] = df["id"].apply(lambda tid: overrides.get(str(tid), {}).get("enabled", True))
        if q:
            df = df[df.apply(lambda r: q.lower() in str(r.id).lower() or q.lower() in str(r.name).lower(), axis=1)]
        if not show_disabled:
            df = df[df.enabled]

        st.write(f"Showing {len(df)} tenant(s)")

        for _, row in df.sort_values("created_at", ascending=False).iterrows():
            tid = int(row.id)
            name = row.name
            created = row.created_at
            enabled = bool(row.enabled)

            cols = st.columns([4,1,1,1,1])
            with cols[0]:
                # NOTE: changed the id styling so the id text is black (for white background)
                st.markdown(
                    f"<div style='padding:12px;border-radius:10px;background:#ffffff;border:1px solid #e6eef0'>"
                    f"<b>{name}</b> <div style='font-size:12px'>"
                    f"<span style='color:#000;font-weight:600'>id: {tid}</span> "
                    f"<span style='color:#6b7280'>• created: {created}</span></div></div>",
                    unsafe_allow_html=True
                )

            # Download DB with confirmation
            if cols[1].button("Download DB", key=f"dl_{tid}"):
                db_path = get_tenant_db_path(tid)
                if not os.path.exists(db_path):
                    st.warning(f"DB file not found: {db_path}")
                else:
                    with open(db_path, "rb") as f:
                        data = f.read()
                    st.download_button(label=f"Download invoices_{tid}.db", data=data, file_name=os.path.basename(db_path), mime="application/octet-stream", key=f"download_{tid}")
                    log_action(super_user, "DOWNLOAD_DB", f"tenant={tid} path={db_path}")

            # ===== Fixed Enable / Disable flow (two-step Confirm/Cancel) =====
            label = "Disable" if enabled else "Enable"

            # Stage 1: clicking the toggle button marks a pending action
            if cols[2].button(label, key=f"toggle_{tid}"):
                st.session_state["pending_toggle_tid"] = tid
                st.session_state["pending_toggle_action"] = "disable" if enabled else "enable"

            # Stage 2: if pending action matches this tenant, show Confirm/Cancel
            if st.session_state.get("pending_toggle_tid") == tid:
                with cols[2]:
                    st.write(f"Confirm {st.session_state.get('pending_toggle_action')}?")
                    if st.button("Confirm", key=f"confirm_toggle_{tid}"):
                        overrides[str(tid)] = overrides.get(str(tid), {})
                        overrides[str(tid)]["enabled"] = not enabled
                        overrides[str(tid)]["modified_at"] = datetime.datetime.now().isoformat()
                        save_overrides(overrides)
                        log_action(super_user, "TOGGLE_ENABLED", f"tenant={tid} now_enabled={overrides[str(tid)]['enabled']}")
                        # clear pending and rerun
                        del st.session_state["pending_toggle_tid"]
                        del st.session_state["pending_toggle_action"]
                        st.success(f"Tenant {tid} updated.")
                        safe_rerun()
                    if st.button("Cancel", key=f"cancel_toggle_{tid}"):
                        # clear pending without changes
                        if "pending_toggle_tid" in st.session_state:
                            del st.session_state["pending_toggle_tid"]
                        if "pending_toggle_action" in st.session_state:
                            del st.session_state["pending_toggle_action"]
                        safe_rerun()

            # show DB path
            dbp = get_tenant_db_path(tid)
            cols[3].markdown(f"<div style='color:#6b7280;font-size:12px'>`{dbp}`</div>", unsafe_allow_html=True)

            # View users quick
            if cols[4].button("View Users", key=f"viewusers_{tid}"):
                try:
                    users = tenants_auth.list_users(tenant_id=tid)
                    if users:
                        udf = pd.DataFrame(users, columns=["id","username","tenant_id","role","created_at"])
                        st.write(f"Users for tenant `{name}` (id={tid}):")
                        st.dataframe(udf)
                    else:
                        st.info("No users for this tenant.")
                except Exception as e:
                    st.error(f"Could not fetch users: {e}")

# ---------- CREATE TENANT ----------
with tab[1]:
    st.header("Create Tenant + Admin User")
    with st.form("create_tenant_form"):
        new_tenant_name = st.text_input("Tenant name")
        admin_username = st.text_input("Admin username")
        admin_pwd = st.text_input("Admin password", type="password")
        create_sub = st.form_submit_button("Create Tenant and Admin")

    if create_sub:
        if not new_tenant_name or not admin_username or not admin_pwd:
            st.error("All fields required.")
        else:
            try:
                tid, final_name = tenants_auth.create_tenant(new_tenant_name)
                tenants_auth.create_user(admin_username, admin_pwd, tenant_id=tid, role="admin")
                st.success(f"Created tenant '{final_name}' (id={tid}) and admin user '{admin_username}'.")
                log_action(super_user, "CREATE_TENANT", f"tenant_id={tid} tenant_name={final_name} admin_user={admin_username}")
                safe_rerun()
            except Exception as e:
                st.error(f"Error creating tenant or user: {e}")

# ---------- USERS & PASSWORDS ----------
with tab[2]:
    st.header("Users & Password Reset")
    try:
        tenants = tenants_auth.list_tenants()
    except Exception as e:
        tenants = []
        st.error(f"Could not load tenants: {e}")

    tenant_opts = {str(t[0]): t[1] for t in tenants}
    choice = st.selectbox("Select tenant", ["-- choose tenant id --"] + [f"{k}: {v}" for k, v in tenant_opts.items()])
    selected_tid = None
    if choice and choice != "-- choose tenant id --":
        selected_tid = int(choice.split(":")[0])

    if selected_tid:
        try:
            users = tenants_auth.list_users(tenant_id=selected_tid)
        except Exception as e:
            st.error(f"Could not load users: {e}")
            users = []
        if users:
            users_df = pd.DataFrame(users, columns=["id","username","tenant_id","role","created_at"])
            st.dataframe(users_df)
        else:
            st.info("No users found for this tenant.")

        with st.form("reset_pw_form"):
            uname = st.selectbox("Select username to reset", [u[1] for u in users] if users else [])
            newpw = st.text_input("New password (for selected user)", type="password")
            if st.form_submit_button("Reset password for user"):
                if not uname or not newpw:
                    st.error("Select user and provide new password.")
                else:
                    try:
                        ok = tenants_auth.reset_user_password(uname, newpw)
                        if ok:
                            st.success(f"Password reset for {uname}.")
                            log_action(super_user, "RESET_PASSWORD", f"tenant={selected_tid} username={uname}")
                        else:
                            st.error("Update reported no rows changed - check username.")
                    except Exception as e:
                        st.error(f"Error resetting password: {e}")

# ---------- OVERRIDES ----------
with tab[3]:
    st.header("Overrides (Enable/Disable tenants + notes)")
    ov_df = []
    for tid_s, meta in overrides.items():
        ov_df.append({"tenant_id": int(tid_s), "enabled": meta.get("enabled", True), "note": meta.get("note",""), "modified_at": meta.get("modified_at","")})
    if ov_df:
        st.dataframe(pd.DataFrame(ov_df))
    else:
        st.info("No overrides configured.")

    with st.form("override_form"):
        tid_in = st.text_input("Tenant id to set override")
        enabled_in = st.checkbox("Enabled?", value=True)
        note_in = st.text_input("Note (optional)")
        set_ok = st.form_submit_button("Save override")
    if set_ok:
        if not tid_in or not tid_in.isnumeric():
            st.error("Enter numeric tenant id.")
        else:
            overrides[tid_in] = {"enabled": enabled_in, "note": note_in or "", "modified_at": datetime.datetime.now().isoformat()}
            save_overrides(overrides)
            log_action(super_user, "SET_OVERRIDE", f"tenant={tid_in} enabled={enabled_in} note={note_in}")
            st.success("Override saved.")
            safe_rerun()

# ---------- AUDIT LOG ----------
with tab[4]:
    st.header("Audit Log (recent)")
    if os.path.exists(AUDIT_LOG):
        with open(AUDIT_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()[-500:]
        st.text_area("Recent Audit Entries", value="".join(lines[::-1]), height=320)
        with open(AUDIT_LOG, "rb") as f:
            data = f.read()
        st.download_button("Download full audit log", data=data, file_name=os.path.basename(AUDIT_LOG))
    else:
        st.info("No audit log yet.")

# footer
st.markdown("---")
st.markdown(
    """
    **Security notes**
    - This Super Admin uses environment variable credentials. Do **not** use defaults in production.
    - Downloading raw DB files exposes tenant data—limit access to trusted admins and use secure transport (HTTPS).
    - Overrides are stored in `super_admin_overrides.json`. Integrate overrides into your tenancy checks where appropriate.
    - Consider adding 2FA, IP restrictions, and proper audit forwarding for production.
    """
)
