from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field


DATA_ROOT = Path(os.environ.get("ORBIT_HUB_DATA", "/data")).resolve()
DB_PATH = DATA_ROOT / "orbit-hub.sqlite3"
UPLOAD_ROOT = DATA_ROOT / "uploads"
MODEL_ROOT = DATA_ROOT / "models"
SETUP_TOKEN_PATH = DATA_ROOT / "initial-admin-token"
CHUNK_SIZE = 8 * 1024 * 1024
MAX_UPLOAD_BYTES = int(os.environ.get("ORBIT_MAX_UPLOAD_BYTES", str(10 * 1024**3)))
MAX_TOTAL_BYTES = int(os.environ.get("ORBIT_MAX_TOTAL_BYTES", str(40 * 1024**3)))
OPEN_REGISTRATION = os.environ.get("ORBIT_OPEN_REGISTRATION", "1") == "1"
COOKIE_SECURE = os.environ.get("ORBIT_COOKIE_SECURE", "1") == "1"
SESSION_SECONDS = 7 * 24 * 60 * 60
ALLOWED_EXTENSIONS = {".pt", ".gguf", ".zip"}
PASSWORD = PasswordHasher(time_cost=2, memory_cost=19_456, parallelism=1)
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
ID_RE = re.compile(r"^[a-f0-9]{32}$")

for directory in (DATA_ROOT, UPLOAD_ROOT, MODEL_ROOT):
    directory.mkdir(parents=True, exist_ok=True)


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(DB_PATH, timeout=20)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def now() -> int:
    return int(time.time())


def initialize() -> None:
    with db() as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS users(
              id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
              role TEXT NOT NULL CHECK(role IN ('user','admin')), created_at INTEGER NOT NULL,
              disabled INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS sessions(
              token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, csrf TEXT NOT NULL,
              expires_at INTEGER NOT NULL, created_at INTEGER NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS models(
              id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL, filename TEXT NOT NULL,
              size INTEGER NOT NULL, sha256 TEXT NOT NULL, preset TEXT NOT NULL,
              parameters INTEGER NOT NULL, description TEXT NOT NULL, status TEXT NOT NULL,
              received_bytes INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL,
              reviewed_at INTEGER, reviewer_id TEXT, review_notes TEXT NOT NULL DEFAULT '',
              FOREIGN KEY(user_id) REFERENCES users(id), FOREIGN KEY(reviewer_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS chunks(
              model_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, size INTEGER NOT NULL,
              PRIMARY KEY(model_id, chunk_index), FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS audit(
              id INTEGER PRIMARY KEY AUTOINCREMENT, actor_id TEXT, action TEXT NOT NULL,
              target_id TEXT, detail TEXT NOT NULL, created_at INTEGER NOT NULL
            );
            """
        )
        admins = connection.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
    if admins == 0 and not SETUP_TOKEN_PATH.exists():
        SETUP_TOKEN_PATH.write_text(secrets.token_urlsafe(32), encoding="utf-8")
        os.chmod(SETUP_TOKEN_PATH, 0o600)
        print("Orbit Hub initial admin token was created. Run: docker compose exec hub cat /data/initial-admin-token", flush=True)


initialize()
app = FastAPI(title="Orbit Hub", docs_url=None, redoc_url=None)
_login_attempts: dict[str, list[int]] = {}


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=200)


class SetupRequest(Credentials):
    setup_token: str = Field(min_length=20, max_length=200)


class UploadRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    filename: str = Field(min_length=1, max_length=180)
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    preset: str = Field(default="custom", max_length=20)
    parameters: int = Field(default=0, ge=0)
    description: str = Field(default="", max_length=2000)


class ReviewRequest(BaseModel):
    decision: str
    notes: str = Field(default="", max_length=2000)


def clean_email(value: str) -> str:
    email = value.strip().lower()
    if not EMAIL_RE.fullmatch(email):
        raise HTTPException(400, "Invalid email address")
    return email


def safe_filename(value: str) -> str:
    name = Path(value).name
    if name != value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,179}", name):
        raise HTTPException(400, "Invalid filename")
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only .pt, .gguf, and .zip model packages are accepted")
    return name


def audit(connection: sqlite3.Connection, actor: str | None, action: str, target: str | None, detail: Any = "") -> None:
    connection.execute(
        "INSERT INTO audit(actor_id,action,target_id,detail,created_at) VALUES(?,?,?,?,?)",
        (actor, action, target, json.dumps(detail, ensure_ascii=False), now()),
    )


def issue_session(connection: sqlite3.Connection, user_id: str) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(24)
    connection.execute("DELETE FROM sessions WHERE expires_at < ?", (now(),))
    connection.execute(
        "INSERT INTO sessions(token_hash,user_id,csrf,expires_at,created_at) VALUES(?,?,?,?,?)",
        (hashlib.sha256(token.encode()).hexdigest(), user_id, csrf, now() + SESSION_SECONDS, now()),
    )
    return token, csrf


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        "__Host-orbit_session" if COOKIE_SECURE else "orbit_session",
        token, max_age=SESSION_SECONDS, secure=COOKIE_SECURE, httponly=True,
        samesite="strict", path="/",
    )


def current_user(
    authorization: str | None = Header(default=None),
    secure_cookie: str | None = Cookie(default=None, alias="__Host-orbit_session"),
    dev_cookie: str | None = Cookie(default=None, alias="orbit_session"),
) -> dict[str, Any]:
    token = authorization[7:].strip() if authorization and authorization.startswith("Bearer ") else (secure_cookie or dev_cookie)
    if not token:
        raise HTTPException(401, "Login required")
    digest = hashlib.sha256(token.encode()).hexdigest()
    with db() as connection:
        row = connection.execute(
            "SELECT u.*,s.csrf,s.expires_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=?",
            (digest,),
        ).fetchone()
    if not row or row["expires_at"] < now() or row["disabled"]:
        raise HTTPException(401, "Session expired")
    return dict(row) | {"session_token_hash": digest, "using_bearer": bool(authorization)}


def require_csrf(request: Request, user: dict[str, Any]) -> None:
    if not user["using_bearer"] and request.method not in {"GET", "HEAD", "OPTIONS"}:
        if not secrets.compare_digest(request.headers.get("X-CSRF-Token", ""), user["csrf"]):
            raise HTTPException(403, "CSRF verification failed")


def admin(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if user["role"] != "admin":
        raise HTTPException(403, "Administrator access required")
    return user


def public_user(row: dict[str, Any]) -> dict[str, Any]:
    return {"id": row["id"], "email": row["email"], "role": row["role"], "created_at": row["created_at"]}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return SERVER_PAGE


@app.get("/health")
def health() -> dict[str, Any]:
    return {"name": "orbit-hub", "status": "ok", "training": False}


@app.get("/api/setup/status")
def setup_status() -> dict[str, Any]:
    with db() as connection:
        initialized = connection.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1").fetchone() is not None
    return {"initialized": initialized, "open_registration": OPEN_REGISTRATION, "max_upload_bytes": MAX_UPLOAD_BYTES}


@app.post("/api/setup")
def setup(payload: SetupRequest, response: Response) -> dict[str, Any]:
    with db() as connection:
        if connection.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1").fetchone():
            raise HTTPException(409, "Administrator already initialized")
        expected = SETUP_TOKEN_PATH.read_text(encoding="utf-8").strip() if SETUP_TOKEN_PATH.exists() else ""
        if not expected or not secrets.compare_digest(expected, payload.setup_token):
            raise HTTPException(403, "Invalid setup token")
        user_id = secrets.token_hex(16)
        email = clean_email(payload.email)
        connection.execute(
            "INSERT INTO users(id,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
            (user_id, email, PASSWORD.hash(payload.password), "admin", now()),
        )
        token, csrf = issue_session(connection, user_id)
        audit(connection, user_id, "admin_initialized", user_id)
    SETUP_TOKEN_PATH.unlink(missing_ok=True)
    set_session_cookie(response, token)
    return {"token": token, "csrf_token": csrf, "user": {"id": user_id, "email": email, "role": "admin"}}


@app.post("/api/register")
def register(payload: Credentials, response: Response) -> dict[str, Any]:
    if not OPEN_REGISTRATION:
        raise HTTPException(403, "Registration is closed")
    with db() as connection:
        if connection.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1").fetchone() is None:
            raise HTTPException(409, "Initialize the administrator before creating user accounts")
    email = clean_email(payload.email)
    user_id = secrets.token_hex(16)
    try:
        with db() as connection:
            connection.execute(
                "INSERT INTO users(id,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                (user_id, email, PASSWORD.hash(payload.password), "user", now()),
            )
            token, csrf = issue_session(connection, user_id)
            audit(connection, user_id, "registered", user_id)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "Account already exists") from exc
    set_session_cookie(response, token)
    return {"token": token, "csrf_token": csrf, "user": {"id": user_id, "email": email, "role": "user"}}


@app.post("/api/login")
def login(payload: Credentials, request: Request, response: Response) -> dict[str, Any]:
    remote = request.client.host if request.client else "unknown"
    recent = [stamp for stamp in _login_attempts.get(remote, []) if stamp > now() - 300]
    if len(recent) >= 8:
        raise HTTPException(429, "Too many login attempts; try again later")
    email = clean_email(payload.email)
    with db() as connection:
        row = connection.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        try:
            if not row or row["disabled"]:
                raise VerifyMismatchError()
            PASSWORD.verify(row["password_hash"], payload.password)
        except VerifyMismatchError as exc:
            recent.append(now())
            _login_attempts[remote] = recent
            audit(connection, None, "login_failed", None, {"email": email})
            raise HTTPException(401, "Email or password is incorrect") from exc
        token, csrf = issue_session(connection, row["id"])
        audit(connection, row["id"], "login", row["id"])
        user = public_user(dict(row))
    _login_attempts.pop(remote, None)
    set_session_cookie(response, token)
    return {"token": token, "csrf_token": csrf, "user": user}


@app.get("/api/me")
def me(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    return public_user(user)


@app.post("/api/logout")
def logout(request: Request, response: Response, user: dict[str, Any] = Depends(current_user)) -> dict[str, bool]:
    require_csrf(request, user)
    with db() as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash=?", (user["session_token_hash"],))
        audit(connection, user["id"], "logout", user["id"])
    response.delete_cookie("__Host-orbit_session" if COOKIE_SECURE else "orbit_session", path="/")
    return {"ok": True}


def model_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["chunks"] = (item["size"] + CHUNK_SIZE - 1) // CHUNK_SIZE
    return item


@app.post("/api/uploads")
def create_upload(payload: UploadRequest, request: Request, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_csrf(request, user)
    if payload.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Model exceeds this server's {MAX_UPLOAD_BYTES} byte upload limit")
    usage = sum(path.stat().st_size for path in MODEL_ROOT.glob("*") if path.is_file())
    free = shutil.disk_usage(DATA_ROOT).free
    if usage + payload.size > MAX_TOTAL_BYTES or free < payload.size + 512 * 1024**2:
        raise HTTPException(507, "Server storage quota is not sufficient for this model")
    filename = safe_filename(payload.filename)
    model_id = secrets.token_hex(16)
    (UPLOAD_ROOT / model_id).mkdir(mode=0o700)
    with db() as connection:
        connection.execute(
            "INSERT INTO models(id,user_id,name,filename,size,sha256,preset,parameters,description,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (model_id, user["id"], payload.name.strip(), filename, payload.size, payload.sha256,
             payload.preset, payload.parameters, payload.description, "uploading", now()),
        )
        audit(connection, user["id"], "upload_created", model_id, {"size": payload.size, "filename": filename})
    return {"id": model_id, "chunk_size": CHUNK_SIZE, "chunks": (payload.size + CHUNK_SIZE - 1) // CHUNK_SIZE}


@app.put("/api/uploads/{model_id}/chunks/{index}")
async def upload_chunk(model_id: str, index: int, request: Request, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_csrf(request, user)
    if not ID_RE.fullmatch(model_id) or index < 0:
        raise HTTPException(400, "Invalid upload")
    with db() as connection:
        row = connection.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        if not row or (row["user_id"] != user["id"] and user["role"] != "admin") or row["status"] != "uploading":
            raise HTTPException(404, "Upload not found")
        chunks = (row["size"] + CHUNK_SIZE - 1) // CHUNK_SIZE
        if index >= chunks:
            raise HTTPException(400, "Chunk index out of range")
        expected = CHUNK_SIZE if index < chunks - 1 else row["size"] - index * CHUNK_SIZE
    body = await request.body()
    if len(body) != expected:
        raise HTTPException(400, f"Expected {expected} bytes for this chunk")
    folder = UPLOAD_ROOT / model_id
    temporary = folder / f"{index}.tmp"
    final = folder / f"{index}.part"
    temporary.write_bytes(body)
    os.chmod(temporary, 0o600)
    os.replace(temporary, final)
    with db() as connection:
        connection.execute("INSERT OR REPLACE INTO chunks(model_id,chunk_index,size) VALUES(?,?,?)", (model_id, index, len(body)))
        received = connection.execute("SELECT COALESCE(SUM(size),0) FROM chunks WHERE model_id=?", (model_id,)).fetchone()[0]
        connection.execute("UPDATE models SET received_bytes=? WHERE id=?", (received, model_id))
    return {"received_bytes": received, "total_bytes": row["size"]}


@app.post("/api/uploads/{model_id}/complete")
def complete_upload(model_id: str, request: Request, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_csrf(request, user)
    if not ID_RE.fullmatch(model_id):
        raise HTTPException(400, "Invalid upload")
    with db() as connection:
        row = connection.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        if not row or row["user_id"] != user["id"] or row["status"] != "uploading":
            raise HTTPException(404, "Upload not found")
    folder = UPLOAD_ROOT / model_id
    chunks = (row["size"] + CHUNK_SIZE - 1) // CHUNK_SIZE
    output = MODEL_ROOT / f"{model_id}{Path(row['filename']).suffix.lower()}.tmp"
    digest = hashlib.sha256()
    written = 0
    try:
        with output.open("wb") as destination:
            for index in range(chunks):
                part = folder / f"{index}.part"
                if not part.is_file():
                    raise HTTPException(409, f"Chunk {index} is missing")
                with part.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                        destination.write(block)
                        written += len(block)
        if written != row["size"] or not secrets.compare_digest(digest.hexdigest(), row["sha256"]):
            raise HTTPException(400, "Model checksum verification failed")
        final = output.with_suffix(Path(row["filename"]).suffix.lower())
        os.chmod(output, 0o600)
        os.replace(output, final)
        shutil.rmtree(folder)
        with db() as connection:
            connection.execute("UPDATE models SET status='pending_review',received_bytes=size WHERE id=?", (model_id,))
            connection.execute("DELETE FROM chunks WHERE model_id=?", (model_id,))
            audit(connection, user["id"], "upload_completed", model_id, {"sha256": row["sha256"]})
        return {"id": model_id, "status": "pending_review", "sha256": row["sha256"]}
    finally:
        output.unlink(missing_ok=True)


@app.get("/api/models")
def list_models(user: dict[str, Any] = Depends(current_user)) -> list[dict[str, Any]]:
    with db() as connection:
        rows = connection.execute(
            "SELECT m.*,u.email AS owner_email FROM models m JOIN users u ON u.id=m.user_id WHERE m.status='approved' OR m.user_id=? ORDER BY m.created_at DESC",
            (user["id"],),
        ).fetchall()
    return [model_row(row) for row in rows]


@app.get("/api/admin/models")
def admin_models(user: dict[str, Any] = Depends(admin)) -> list[dict[str, Any]]:
    with db() as connection:
        rows = connection.execute(
            "SELECT m.*,u.email AS owner_email FROM models m JOIN users u ON u.id=m.user_id ORDER BY m.created_at DESC"
        ).fetchall()
    return [model_row(row) for row in rows]


@app.post("/api/admin/models/{model_id}/review")
def review_model(model_id: str, payload: ReviewRequest, request: Request, user: dict[str, Any] = Depends(admin)) -> dict[str, Any]:
    require_csrf(request, user)
    if payload.decision not in {"approved", "rejected"} or not ID_RE.fullmatch(model_id):
        raise HTTPException(400, "Invalid review decision")
    with db() as connection:
        row = connection.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        if not row or row["status"] not in {"pending_review", "approved", "rejected"}:
            raise HTTPException(404, "Reviewable model not found")
        connection.execute(
            "UPDATE models SET status=?,reviewed_at=?,reviewer_id=?,review_notes=? WHERE id=?",
            (payload.decision, now(), user["id"], payload.notes, model_id),
        )
        audit(connection, user["id"], f"model_{payload.decision}", model_id, {"notes": payload.notes})
    return {"id": model_id, "status": payload.decision}


@app.get("/api/admin/users")
def admin_users(user: dict[str, Any] = Depends(admin)) -> list[dict[str, Any]]:
    with db() as connection:
        rows = connection.execute("SELECT id,email,role,created_at,disabled FROM users ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


@app.get("/api/models/{model_id}/download")
def download_model(model_id: str, user: dict[str, Any] = Depends(current_user)) -> FileResponse:
    if not ID_RE.fullmatch(model_id):
        raise HTTPException(400, "Invalid model")
    with db() as connection:
        row = connection.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
    if not row or (row["status"] != "approved" and row["user_id"] != user["id"] and user["role"] != "admin"):
        raise HTTPException(404, "Model not found")
    path = next(MODEL_ROOT.glob(f"{model_id}.*"), None)
    if not path or not path.is_file():
        raise HTTPException(404, "Model file is unavailable")
    return FileResponse(path, filename=row["filename"], media_type="application/octet-stream")


SERVER_PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Orbit Hub</title><style>
:root{font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#202633;background:#edf1f7}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 85% 0,#e8e0ff,transparent 34%),#edf1f7}.wrap{max-width:1050px;margin:auto;padding:48px 20px}.brand{display:flex;align-items:center;gap:12px;margin-bottom:28px}.brand b{font-size:22px}.card{background:rgba(255,255,255,.78);border:1px solid #fff;border-radius:18px;padding:22px;box-shadow:0 18px 60px rgba(31,45,73,.1);backdrop-filter:blur(24px);margin-bottom:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}input,button,textarea{font:inherit;width:100%;padding:10px 12px;border-radius:10px;border:1px solid #d9deea;margin:5px 0 10px}button{border:0;background:#2168f3;color:white;font-weight:700;cursor:pointer}button.secondary{background:#e7ecf4;color:#202633}.muted{color:#687386;font-size:13px}.row{padding:12px 0;border-bottom:1px solid #e3e7ee}.badge{display:inline-block;padding:3px 8px;border-radius:99px;background:#e8eefc;color:#2168f3;font-size:11px}.hide{display:none}@media(max-width:720px){.grid{grid-template-columns:1fr}}
</style></head><body><main class="wrap"><div class="brand"><b>Orbit Hub</b><span class="badge">No server-side training</span></div><div id="message" class="card muted">Loading…</div><section id="setup" class="card hide"><h2>Initialize administrator / 初始化管理员</h2><input id="setupToken" placeholder="One-time setup token"><input id="setupEmail" type="email" placeholder="Admin email"><input id="setupPassword" type="password" placeholder="Password (12+ characters)"><button onclick="initializeAdmin()">Create administrator</button></section><section id="login" class="card hide"><h2>Login / 登录</h2><input id="email" type="email" placeholder="Email"><input id="password" type="password" placeholder="Password"><button onclick="signIn()">Login</button><button class="secondary" onclick="register()">Create user account</button></section><section id="dashboard" class="hide"><div class="grid"><div class="card"><h2>Account</h2><div id="account"></div><button class="secondary" onclick="logout()">Logout</button></div><div class="card"><h2>Server limits</h2><div id="limits" class="muted"></div></div></div><div id="admin" class="card hide"><h2>Administrator review</h2><p class="muted">Uploaded files are never executed or loaded. Approval only changes download visibility.</p><div id="models"></div></div></section></main><script>
let csrf='';const $=id=>document.getElementById(id);function show(id,on=true){$(id).classList.toggle('hide',!on)}function msg(text){$('message').textContent=text}async function api(path,options={}){const r=await fetch(path,{credentials:'same-origin',headers:{'Content-Type':'application/json',...(csrf?{'X-CSRF-Token':csrf}:{})},...options});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Request failed');return d}async function boot(){try{const s=await api('/api/setup/status');$('limits').textContent=`Per-model upload limit: ${(s.max_upload_bytes/1073741824).toFixed(1)} GB`;if(!s.initialized){show('setup');show('login',false);msg('Use the one-time token printed by the server.');return}try{const u=await api('/api/me');loggedIn(u)}catch{show('login');msg('Login to Orbit Hub.')}}catch(e){msg(e.message)}}async function initializeAdmin(){try{const d=await api('/api/setup',{method:'POST',body:JSON.stringify({setup_token:$('setupToken').value,email:$('setupEmail').value,password:$('setupPassword').value})});csrf=d.csrf_token;loggedIn(d.user)}catch(e){msg(e.message)}}async function signIn(){try{const d=await api('/api/login',{method:'POST',body:JSON.stringify({email:$('email').value,password:$('password').value})});csrf=d.csrf_token;loggedIn(d.user)}catch(e){msg(e.message)}}async function register(){try{const d=await api('/api/register',{method:'POST',body:JSON.stringify({email:$('email').value,password:$('password').value})});csrf=d.csrf_token;loggedIn(d.user)}catch(e){msg(e.message)}}async function loggedIn(user){show('setup',false);show('login',false);show('dashboard');$('account').innerHTML=`<b>${user.email}</b><p class="muted">Role: ${user.role}</p>`;msg(user.role==='admin'?'Administrator account active.':'User account active.');if(user.role==='admin'){show('admin');loadModels()}}async function loadModels(){try{const rows=await api('/api/admin/models');$('models').innerHTML=rows.length?rows.map(x=>`<div class="row"><b>${escapeHtml(x.name)}</b> · ${escapeHtml(x.owner_email)}<br><span class="muted">${escapeHtml(x.filename)} · ${(x.size/1073741824).toFixed(2)} GB · SHA-256 ${escapeHtml(x.sha256)}</span><p><span class="badge">${escapeHtml(x.status)}</span></p>${x.status==='pending_review'?`<textarea id="note-${x.id}" placeholder="Review notes"></textarea><button onclick="review('${x.id}','approved')">Approve</button><button class="secondary" onclick="review('${x.id}','rejected')">Reject</button>`:''}</div>`).join(''):'No uploaded models.'}catch(e){msg(e.message)}}async function review(id,decision){try{await api(`/api/admin/models/${id}/review`,{method:'POST',body:JSON.stringify({decision,notes:$(`note-${id}`).value})});loadModels()}catch(e){msg(e.message)}}async function logout(){try{await api('/api/logout',{method:'POST',body:'{}'})}catch{}location.reload()}function escapeHtml(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}boot();
</script></body></html>'''
SERVER_PAGE = SERVER_PAGE.replace("$('account').innerHTML=`<b>${user.email}</b><p class=\"muted\">Role: ${user.role}</p>`;", "$('account').textContent=user.email+' · Role: '+user.role;")
