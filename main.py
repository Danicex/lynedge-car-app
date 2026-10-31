"""
Vehicle inspection API (single file, FastAPI + SQLModel)

Install:  pip install -r requirements.txt
Run:      uvicorn main:app --reload

Access model
  - Form submitters ("users") have NO accounts and need no auth. They can only WRITE.
  - Admins are the only accounts. Admins can READ (and manage admins).

First admin (done locally, never over HTTP):
  python main.py create-admin

Endpoints
  POST /api/vehicle-inspections          PUBLIC   write only (also accepts a trailing slash)
  POST /api/auth/login                   PUBLIC   admin email + password -> 1-hour bearer token
  POST /api/auth/logout                  ADMIN
  GET  /api/auth/me                      ADMIN
  GET  /api/vehicle-inspections          ADMIN    read
  GET  /api/vehicle-inspections/{id}     ADMIN    read
  POST /create_admin                     ADMIN    add another admin
  POST /update_password                  ADMIN    change an admin's password

Database
  Default is a local SQLite file (DB_PATH, default inspections.db).
  Set DATABASE_URL to use another database (e.g. Postgres) without changing code.
"""
import hashlib
import hmac
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, Dict, List, Literal, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Query, Request,  UploadFile, File
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import StringConstraints, field_validator
from sqlalchemy import JSON, Column, Text
from sqlmodel import Field, Session, SQLModel, create_engine, func, select
from starlette.exceptions import HTTPException as StarletteHTTPException
import uuid
import boto3

from dotenv import load_dotenv
load_dotenv()
import logging
logger = logging.getLogger("uvicorn.error")


# ---------------- config ----------------
DB_PATH = os.getenv("DB_PATH", "inspections.db")
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{DB_PATH}"
if DATABASE_URL.startswith("postgres://"):  # some hosts still hand out the old scheme
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]
TOKEN_TTL = timedelta(hours=1)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS") # restrict in production
MAX_FAILED_LOGINS = 5          # per email + IP
LOCKOUT_SECONDS = 15 * 60
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

PNG_PREFIX = "data:image/png;base64,"
IsoDate = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]
OptionalIsoDate = Annotated[str, StringConstraints(pattern=r"^(\d{4}-\d{2}-\d{2})?$")]
VehicleStatus = Literal[
    "Fit for use",
    "Fit after minor repairs",
    "Do not use until safety defects are corrected",
]


def now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------- database tables ----------------
# Timestamps are stored as ISO-8601 strings so databases created by the earlier raw-SQL
# version of this API keep working unchanged.
class Admin(SQLModel, table=True):
    __tablename__ = "admins"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True, max_length=254)
    password_hash: str                                   # scrypt$salt$hash
    token: Optional[str] = Field(default=None, index=True)  # sha256 of the current token
    token_expires_at: Optional[str] = None               # ISO timestamp (UTC)
    created_at: str = Field(default_factory=lambda: now().isoformat())


class CarFormData(SQLModel, table=True):
    __tablename__ = "car_form_data"

    id: Optional[int] = Field(default=None, primary_key=True)

    # page 1 - vehicle details
    territory_name: str
    inspection_date: str
    field_supervisor_name: str
    supervisor_signature: str = Field(sa_column=Column(Text, nullable=False))  # PNG data URL
    vehicle_user_driver: str
    vehicle_registration_no: str
    vehicle_make_model: str
    odometer_km: int

    # pages 1-4 - checklist
    items: List[dict] = Field(sa_column=Column(JSON, nullable=False))
    good_count: int
    needs_attention_count: int

    # page 5 - corrective actions and sign-off
    corrective_actions: Optional[List[dict]] = Field(default_factory=list, sa_column=Column(JSON))
    overall_vehicle_status: str
    driver_acknowledgement_name: str
    driver_acknowledgement_signature: str = Field(sa_column=Column(Text, nullable=False))  # PNG data URL
    supervisor_final_comments: Optional[str] = None
    supervisor_final_signature: str = Field(sa_column=Column(Text, nullable=False))        # PNG data URL
    overall_remarks: Optional[str] = None                # same text as final comments (kept for compatibility)

    submitted_at: Optional[str] = None
    created_at: str


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(title="Vehicle Inspection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
bearer = HTTPBearer(auto_error=False)


# ---------------- request schemas (what the form sends) ----------------
class Item(SQLModel):
    id: str = Field(max_length=10)
    section: str = Field(max_length=5)
    section_title: str = Field(max_length=200)
    item: str = Field(max_length=500)
    status: Literal["Good", "Needs Attention"]
    remarks: str = Field(default="", max_length=2000)


class Summary(SQLModel):
    good: int
    needs_attention: int


class CorrectiveAction(SQLModel):
    defect: str = Field(default="", max_length=1000)
    priority: Literal["", "Urgent", "Normal"] = ""
    action_owner: str = Field(default="", max_length=200)
    target_date: OptionalIsoDate = ""
    completed_verified_by: str = Field(default="", max_length=300)


class InspectionCreate(SQLModel):
    territory_name: str = Field(min_length=1, max_length=200)
    inspection_date: IsoDate
    field_supervisor_name: str = Field(min_length=1, max_length=200)
    supervisor_signature: str = Field(max_length=700_000)
    vehicle_user_driver: str = Field(min_length=1, max_length=200)
    vehicle_registration_no: str = Field(min_length=1, max_length=50)
    vehicle_make_model: str = Field(min_length=1, max_length=200)
    odometer_km: int = Field(ge=0, le=10_000_000)
    items: List[Item] = Field(min_length=1, max_length=200)
    summary: Optional[Summary] = None                    # accepted but ignored: counts are recomputed here
    corrective_actions: List[CorrectiveAction] = Field(default_factory=list, max_length=100)
    overall_vehicle_status: VehicleStatus
    driver_acknowledgement_name: str = Field(min_length=1, max_length=200)
    driver_acknowledgement_signature: str = Field(max_length=700_000)
    supervisor_final_comments: str = Field(default="", max_length=5000)
    supervisor_final_signature: str = Field(max_length=700_000)
    overall_remarks: str = Field(default="", max_length=5000)  # falls back to final comments
    submitted_at: Optional[str] = Field(default=None, max_length=40)

    @field_validator("supervisor_signature", "driver_acknowledgement_signature", "supervisor_final_signature")
    @classmethod
    def must_be_png(cls, v: str) -> str:
        if not v.startswith(PNG_PREFIX) or len(v) <= len(PNG_PREFIX):
            raise ValueError("must be a PNG data URL")
        return v


class LoginRequest(SQLModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class CreateAdminRequest(SQLModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=256)


class UpdatePasswordRequest(SQLModel):
    admin_id: int
    email: str = Field(min_length=3, max_length=254)
    updated_password: str = Field(min_length=8, max_length=256)
    current_password: Optional[str] = Field(default=None, max_length=256)  # required for your own account


# ---------------- errors: the form reads `message`, so always include it ----------------
@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException):
    return JSONResponse(
        {"detail": exc.detail, "message": exc.detail},
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    problems = [
        {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
        for e in exc.errors()
    ]
    message = "; ".join(f'{p["field"]}: {p["message"]}' for p in problems[:5])
    return JSONResponse({"detail": problems, "message": message or "Invalid request"}, status_code=422)


# ---------------- passwords & tokens ----------------
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


def find_admin(session: Session, email: str) -> Optional[Admin]:
    return session.exec(select(Admin).where(func.lower(Admin.email) == email.strip().lower())).first()


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
def require_admin(
    session: SessionDep,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> Admin:
    """Valid, unexpired admin token required."""
    if creds is None:
        raise HTTPException(401, "Missing token", headers={"WWW-Authenticate": "Bearer"})
    admin = session.exec(select(Admin).where(Admin.token == hash_token(creds.credentials))).first()
    if admin is None:
        raise HTTPException(401, "Invalid token", headers={"WWW-Authenticate": "Bearer"})
    if not admin.token_expires_at or datetime.fromisoformat(admin.token_expires_at) < now():
        raise HTTPException(401, "Token expired", headers={"WWW-Authenticate": "Bearer"})
    return admin


AdminDep = Annotated[Admin, Depends(require_admin)]


@app.post("/api/auth/login")
def login(body: LoginRequest, request: Request, session: SessionDep):
    key = _throttle_key(body.email, request)
    _check_locked(key)

    admin = find_admin(session, body.email)
    # always run a password check so unknown emails take as long as known ones
    password_ok = verify_password(body.password, admin.password_hash if admin else _DUMMY_HASH)
    if admin is None or not password_ok:
        _record_failure(key)
        raise HTTPException(401, "Invalid email or password")

    _failed.pop(key, None)
    token = secrets.token_urlsafe(32)
    expires = now() + TOKEN_TTL
    admin.token = hash_token(token)
    admin.token_expires_at = expires.isoformat()
    session.add(admin)
    session.commit()
    return {
        "token": token,
        "token_type": "bearer",
        "expires_at": expires.isoformat(),
        "expires_in": int(TOKEN_TTL.total_seconds()),
        "admin": {"id": admin.id, "email": admin.email},
    }


@app.post("/api/auth/logout")
def logout(admin: AdminDep, session: SessionDep):
    admin.token = None
    admin.token_expires_at = None
    session.add(admin)
    session.commit()
    return {"message": "Logged out"}


@app.get("/api/auth/me")
def me(admin: AdminDep):
    return {"id": admin.id, "email": admin.email, "token_expires_at": admin.token_expires_at}


# ---------------- admin management (ADMIN only) ----------------
# create_admin is not public: an open one would let anyone make themselves an admin.
# The first admin is created locally with `python main.py create-admin`.
@app.post("/create_admin", status_code=201)
def create_admin(body: CreateAdminRequest,  session: SessionDep):
    email = body.email.strip().lower()
    if "@" not in email or " " in email:
        raise HTTPException(422, "Invalid email address")
    if find_admin(session, email):
        raise HTTPException(409, "An admin with this email already exists")
    new = Admin(email=email, password_hash=hash_password(body.password))
    session.add(new)
    session.commit()
    session.refresh(new)
    return {"id": new.id, "email": new.email, "message": "Admin created"}


@app.post("/update_password")
def update_admin_password(body: UpdatePasswordRequest,  session: SessionDep):
    target = find_admin(session, body.email)
    if not target:
        raise HTTPException(404, "Admin not found")
    # changing your own password needs the current one (protects an unattended session)
    if target.id == body.admin_id:
        if not body.current_password or not verify_password(body.current_password, target.password_hash):
            raise HTTPException(403, "Current password is missing or incorrect")
    # new hash + clear the token so every existing session for that admin ends
    target.password_hash = hash_password(body.updated_password)
    target.token = None
    target.token_expires_at = None
    session.add(target)
    session.commit()
    return {"message": "Password updated. That admin must log in again."}


# ---------------- inspections ----------------
# The form posts to ".../vehicle-inspections/" (trailing slash); both spellings are accepted
# so the browser is never sent through a redirect (which breaks cross-origin POSTs).
@app.post("/api/vehicle-inspections", status_code=201)  # PUBLIC: write only
@app.post("/api/vehicle-inspections/", status_code=201, include_in_schema=False)
def create_inspection(data: InspectionCreate, session: SessionDep):
    good = sum(1 for i in data.items if i.status == "Good")
    fields = data.model_dump(exclude={"summary"})
    fields["overall_remarks"] = data.overall_remarks or data.supervisor_final_comments
    row = CarFormData(
        **fields,
        good_count=good,
        needs_attention_count=len(data.items) - good,
        created_at=now().isoformat(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return {"id": row.id, "message": "Inspection submitted"}  # no form data echoed back


@app.get("/api/vehicle-inspections")  # ADMIN: read
@app.get("/api/vehicle-inspections/", include_in_schema=False)
def list_inspections(
    _admin: AdminDep,
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    rows = session.exec(select(CarFormData).order_by(CarFormData.id.desc()).offset(offset).limit(limit)).all()
    total = session.exec(select(func.count()).select_from(CarFormData)).one()
    return {"total": total, "results": rows}


@app.get("/api/vehicle-inspections/{inspection_id}")  # ADMIN: read
def get_inspection(inspection_id: int, _admin: AdminDep, session: SessionDep):
    row = session.get(CarFormData, inspection_id)
    if not row:
        raise HTTPException(404, "Not found")
    return row


# ---------------- FILE UPLOAD ----------------

r2 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    print(bool(R2_ACCOUNT_ID), bool(R2_ACCESS_KEY_ID), bool(R2_SECRET_ACCESS_KEY), bool(R2_BUCKET_NAME))
    if file.content_type != "video/mp4":
        raise HTTPException(
            status_code=400,
            detail="Only MP4 videos are allowed.",
        )

    video_id = f"{uuid.uuid4()}.mp4"

    try:
        r2.upload_fileobj(
            file.file,
            R2_BUCKET_NAME,
            video_id,
            ExtraArgs={"ContentType": "video/mp4"},
        )

        return {
            "message": "Video uploaded successfully",
            "video_id": video_id,
        }

    except Exception as e:
        logger.exception("R2 upload failed")  # prints full traceback in the uvicorn console
        raise HTTPException(status_code=500, detail="Failed to upload video.") from e

    finally:
        await file.close()
        
# ---------------- CLI: python main.py create-admin ----------------
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
        create_db_and_tables()
        with Session(engine) as s:
            if find_admin(s, email):
                sys.exit("An admin with that email already exists.")
            s.add(Admin(email=email, password_hash=hash_password(pw)))
            s.commit()
        print(f"Admin {email} created in {DATABASE_URL.split('@')[-1]}.")
    else:
        print("Usage: python main.py create-admin     (run the API with: uvicorn main:app --reload)")