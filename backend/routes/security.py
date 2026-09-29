"""Password hashing (PBKDF2-SHA256, standard library) and JWT bearer tokens (PyJWT)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.db import get_db
from backend.database.models import Document, User

ITERATIONS = 240_000
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt_b64), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(hash_b64))
    except (ValueError, TypeError):
        return False


def create_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "iat": now, "exp": now + timedelta(hours=settings.TOKEN_EXPIRE_HOURS)}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


def _unauthorized(detail: str = "Not authenticated"):
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail,
                         headers={"WWW-Authenticate": "Bearer"})


def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
                     db: Session = Depends(get_db)) -> User:
    if creds is None:
        raise _unauthorized()
    try:
        payload = jwt.decode(creds.credentials, settings.SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Session expired, please log in again")
    except jwt.PyJWTError:
        raise _unauthorized("Invalid token")
    user = db.get(User, int(payload.get("sub", 0)))
    if user is None:
        raise _unauthorized("User no longer exists")
    return user


def get_owned_document(document_id: int, user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)) -> Document:
    doc = db.get(Document, document_id)
    if doc is None or doc.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


def require_completed(doc: Document) -> Document:
    if doc.status != "completed":
        raise HTTPException(status_code=409, detail=f"Document analysis is not complete (status: {doc.status})")
    return doc
