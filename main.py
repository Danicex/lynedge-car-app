"""
Vehicle inspection API (single file, FastAPI + SQLite)

Install:  pip install -r requirements.txt
Run:      uvicorn server:app --reload

Access model
  - Form submitters ("users") have NO accounts and need no auth. They can only WRITE.
  - Admins are the only accounts. Admins can READ (and manage admins).

First admin (done locally, never over HTTP):
  python server.py create-admin

Endpoints
  POST /api/vehicle-inspections          PUBLIC   write only
  POST /api/auth/login                   PUBLIC   admin email + password -> 1-hour bearer token
  POST /api/auth/logout                  ADMIN
  GET  /api/auth/me                      ADMIN
  GET  /api/vehicle-inspections          ADMIN    read
  GET  /api/vehicle-inspections/{id}     ADMIN    read
  POST /create_admin                     ADMIN    add another admin
  POST /update_password                  ADMIN    change an admin's password
"""
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# ---------------- config ----------------
DB_PATH = os.getenv("DB_PATH", "inspections.db")
TOKEN_TTL = timedelta(hours=1)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")  # restrict in production
MAX_FAILED_LOGINS = 5          # per email + IP
LOCKOUT_SECONDS = 15 * 60

app = FastAPI(title="Vehicle Inspection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
bearer = HTTPBearer(auto_error=False)


# ---------------- helpers ----------------
def now() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    """salted scrypt (stdlib): scrypt$<salt>$<hash>"""
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        h = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1, dklen=32)
        return hmac.compare_digest(h.hex(), hash_hex)
    except Exception:
        return False


def hash_token(token: str) -> str:
    # only the hash is stored, so a leaked DB does not leak live tokens
    return hashlib.sha256(token.encode()).hexdigest()


# Checked when the email is unknown, so response time doesn't reveal which emails exist.
_DUMMY_HASH = hash_password("dummy-password-for-timing")


# ---------------- database ----------------
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
def init_db():
    with db() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,   -- scrypt$salt$hash
                token TEXT,                    -- sha256 of the current token
                token_expires_at TEXT,         -- ISO timestamp (UTC)
                created_at TEXT NOT NULL
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS car_form_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                territory_name TEXT NOT NULL,
                inspection_date TEXT NOT NULL,
                field_supervisor_name TEXT NOT NULL,
                supervisor_signature TEXT NOT NULL,   -- PNG data URL
                vehicle_user_driver TEXT NOT NULL,
                vehicle_registration_no TEXT NOT NULL,
                vehicle_make_model TEXT NOT NULL,
                odometer_km INTEGER NOT NULL,
                items TEXT NOT NULL,                  -- JSON list of checklist rows
                good_count INTEGER NOT NULL,
                needs_attention_count INTEGER NOT NULL,
                overall_remarks TEXT,
                submitted_at TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        # carry over admins from the earlier "users" table, if there is one (they log in again)
        old = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'").fetchone()
        if old:
            cols = [r["name"] for r in c.execute("PRAGMA table_info(users)")]
            if "password_hash" in cols and "is_admin" in cols:
                c.execute(
                    """INSERT OR IGNORE INTO admins (email, password_hash, created_at)
                       SELECT lower(email), password_hash, ? FROM users
                       WHERE is_admin = 1 AND password_hash IS NOT NULL""",
                    (now().isoformat(),),
                )


# ---------------- schemas ----------------
class Item(BaseModel):
    id: str
    section: str
    section_title: str
    item: str
    status: str = Field(pattern="^(Good|Needs Attention)$")
    remarks: str = ""


class Summary(BaseModel):
    good: int
    needs_attention: int


class InspectionIn(BaseModel):
    territory_name: str = Field(min_length=1, max_length=200)
    inspection_date: str = Field(min_length=1, max_length=20)
    field_supervisor_name: str = Field(min_length=1, max_length=200)
    supervisor_signature: str = Field(min_length=1, max_length=700_000)
    vehicle_user_driver: str = Field(min_length=1, max_length=200)
    vehicle_registration_no: str = Field(min_length=1, max_length=50)
    vehicle_make_model: str = Field(min_length=1, max_length=200)
    odometer_km: int = Field(ge=0)
    items: List[Item] = Field(min_length=1, max_length=200)
    summary: Optional[Summary] = None
    overall_remarks: str = Field(default="", max_length=5000)
    submitted_at: Optional[str] = None


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class CreateAdminRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=256)


class UpdatePasswordRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    updated_password: str = Field(min_length=8, max_length=256)
    current_password: Optional[str] = Field(default=None, max_length=256)  # required for your own account


def row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["items"] = json.loads(d["items"])
    return d


# ---------------- login throttling (in-memory, per process) ----------------
_failed: Dict[Tuple[str, str], Tuple[int, float]] = {}  # (email, ip) -> (count, first_failure_time)


def _throttle_key(email: str, request: Request) -> Tuple[str, str]:
    ip = request.client.host if request.client else "unknown"
    return (email.strip().lower(), ip)


def _check_locked(key: Tuple[str, str]):
    count, first = _failed.get(key, (0, 0.0))
    if count >= MAX_FAILED_LOGINS:
        remaining = int(first + LOCKOUT_SECONDS - time.time())
        if remaining > 0:
            raise HTTPException(
                429, "Too many failed attempts. Try again later.", headers={"Retry-After": str(remaining)}
            )
        _failed.pop(key, None)  # lockout window has passed


def _record_failure(key: Tuple[str, str]):
    count, first = _failed.get(key, (0, time.time()))
    _failed[key] = (count + 1, first if count else time.time())


# ---------------- auth ----------------
def require_admin(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> dict:
    """Valid, unexpired admin token required."""
    if creds is None:
        raise HTTPException(401, "Missing token", headers={"WWW-Authenticate": "Bearer"})
    with db() as c:
        admin = c.execute("SELECT * FROM admins WHERE token = ?", (hash_token(creds.credentials),)).fetchone()
    if admin is None:
        raise HTTPException(401, "Invalid token", headers={"WWW-Authenticate": "Bearer"})
    if datetime.fromisoformat(admin["token_expires_at"]) < now():
        raise HTTPException(401, "Token expired", headers={"WWW-Authenticate": "Bearer"})
    return dict(admin)


@app.post("/api/auth/login")
def login(body: LoginRequest, request: Request):
    key = _throttle_key(body.email, request)
    _check_locked(key)

    with db() as c:
        admin = c.execute("SELECT * FROM admins WHERE lower(email) = lower(?)", (body.email.strip(),)).fetchone()
        # always run a password check so unknown emails take as long as known ones
        password_ok = verify_password(body.password, admin["password_hash"] if admin else _DUMMY_HASH)
        if admin is None or not password_ok:
            _record_failure(key)
            raise HTTPException(401, "Invalid email or password")

        _failed.pop(key, None)
        token = secrets.token_urlsafe(32)
        expires = now() + TOKEN_TTL
        c.execute(
            "UPDATE admins SET token = ?, token_expires_at = ? WHERE id = ?",
            (hash_token(token), expires.isoformat(), admin["id"]),
        )
    return {
        "token": token,
        "token_type": "bearer",
        "expires_at": expires.isoformat(),
        "expires_in": int(TOKEN_TTL.total_seconds()),
        "admin": {"id": admin["id"], "email": admin["email"]},
    }


@app.post("/api/auth/logout")
def logout(admin: dict = Depends(require_admin)):
    with db() as c:
        c.execute("UPDATE admins SET token = NULL, token_expires_at = NULL WHERE id = ?", (admin["id"],))
    return {"message": "Logged out"}


@app.get("/api/auth/me")
def me(admin: dict = Depends(require_admin)):
    return {"id": admin["id"], "email": admin["email"], "token_expires_at": admin["token_expires_at"]}


# ---------------- admin management (ADMIN only) ----------------
# create_admin is not public: an open one would let anyone make themselves an admin.
# The first admin is created locally with `python server.py create-admin`.
@app.post("/create_admin", status_code=201)
def create_admin(body: CreateAdminRequest):
    email = body.email.strip().lower()
    if "@" not in email or " " in email:
        raise HTTPException(422, "Invalid email address")
    with db() as c:
        if c.execute("SELECT 1 FROM admins WHERE lower(email) = ?", (email,)).fetchone():
            raise HTTPException(409, "An admin with this email already exists")
        cur = c.execute(
            "INSERT INTO admins (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, hash_password(body.password), now().isoformat()),
        )
    return {"id": cur.lastrowid, "email": email, "message": "Admin created"}


@app.post("/update_password")
def update_admin_password(body: UpdatePasswordRequest):
    email = body.email.strip().lower()
    with db() as c:
        target = c.execute("SELECT * FROM admins WHERE lower(email) = ?", (email,)).fetchone()
        if not target:
            raise HTTPException(404, "Admin not found")
        # changing your own password needs the current one (protects an unattended session)
        # if target["id"] == admin["id"]:
        #     if not body.current_password or not verify_password(body.current_password, target["password_hash"]):
        #         raise HTTPException(403, "Current password is missing or incorrect")
        # new hash + clear the token so every existing session for that admin ends
        c.execute(
            "UPDATE admins SET password_hash = ?, token = NULL, token_expires_at = NULL WHERE id = ?",
            (hash_password(body.updated_password), target["id"]),
        )
    return {"message": "Password updated. That admin must log in again."}


# ---------------- inspections ----------------
@app.post("/api/vehicle-inspections", status_code=201)  # PUBLIC: write only
def create_inspection(data: InspectionIn):
    good = sum(1 for i in data.items if i.status == "Good")
    need = len(data.items) - good
    with db() as c:
        cur = c.execute(
            """INSERT INTO car_form_data
               (territory_name, inspection_date, field_supervisor_name, supervisor_signature,
                vehicle_user_driver, vehicle_registration_no, vehicle_make_model, odometer_km,
                items, good_count, needs_attention_count, overall_remarks, submitted_at, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data.territory_name, data.inspection_date, data.field_supervisor_name,
                data.supervisor_signature, data.vehicle_user_driver, data.vehicle_registration_no,
                data.vehicle_make_model, data.odometer_km,
                json.dumps([i.model_dump() for i in data.items]),
                good, need, data.overall_remarks, data.submitted_at, now().isoformat(),
            ),
        )
    return {"id": cur.lastrowid, "message": "Inspection submitted"}  # no form data echoed back


@app.get("/api/vehicle-inspections")  # ADMIN: read
def list_inspections(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    # _admin: dict = Depends(require_admin),
):
    with db() as c:
        rows = c.execute(
            "SELECT * FROM car_form_data ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        total = c.execute("SELECT COUNT(*) FROM car_form_data").fetchone()[0]
    return {"total": total, "results": [row_to_dict(r) for r in rows]}


@app.get("/api/vehicle-inspections/{inspection_id}")  # ADMIN: read
def get_inspection(inspection_id: int, _admin: dict = Depends(require_admin)):
    with db() as c:
        row = c.execute("SELECT * FROM car_form_data WHERE id = ?", (inspection_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Not found")
    return row_to_dict(row)


# ---------------- CLI: python server.py create-admin ----------------
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "create-admin":
        import getpass

        email = input("Admin email: ").strip().lower()
        if "@" not in email or " " in email:
            sys.exit("Invalid email address.")
        pw = getpass.getpass("Password (min 8 chars): ")
        if len(pw) < 8:
            sys.exit("Password must be at least 8 characters.")
        if pw != getpass.getpass("Confirm password: "):
            sys.exit("Passwords do not match.")
        init_db()  # make sure the tables exist
        try:
            with db() as c:
                c.execute(
                    "INSERT INTO admins (email, password_hash, created_at) VALUES (?, ?, ?)",
                    (email, hash_password(pw), now().isoformat()),
                )
        except sqlite3.IntegrityError:
            sys.exit("An admin with that email already exists.")
        print(f"Admin {email} created in {DB_PATH}.")
    else:
        print("Usage: python server.py create-admin     (run the API with: uvicorn server:app --reload)")