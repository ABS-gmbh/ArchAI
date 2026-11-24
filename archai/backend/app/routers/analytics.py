"""Analytics endpoints with JWT authentication."""

import math
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import settings
from app.schemas.analytics import LoginRequest, LoginResponse
from app.services.analytics_service import get_analytics_data
from app.services.login_limiter import FailedLoginLimiter

router = APIRouter(tags=["analytics"])
security = HTTPBearer()

ALGORITHM = "HS256"
_login_limiter = FailedLoginLimiter()


def _create_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    return jwt.encode(
        {"sub": username, "exp": expire},
        settings.resolve_jwt_secret(),
        algorithm=ALGORITHM,
    )


def _require_configured_auth() -> None:
    """Fail closed when no operator credentials are configured."""
    if not settings.analytics_auth_configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "Analytics authentication is not configured. Set ANALYTICS_USERNAME, "
                "ANALYTICS_PASSWORD and JWT_SECRET in the backend environment."
            ),
        )


def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    _require_configured_auth()
    try:
        payload = jwt.decode(
            credentials.credentials, settings.resolve_jwt_secret(), algorithms=[ALGORITHM]
        )
        username = payload.get("sub")
        # Signed with our secret is not enough: the token must be the operator's.
        if not isinstance(username, str) or not secrets.compare_digest(
            username.encode(), settings.analytics_username.encode()
        ):
            raise HTTPException(status_code=401, detail="Invalid token.")
        return username
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token.") from exc


def _username_key(username: str) -> str:
    """One key per user name however it is spelled, and of bounded size."""
    return "user:" + " ".join(username.split()).casefold()[:128]


@router.post("/analytics/login")
def analytics_login(body: LoginRequest, request: Request):
    _require_configured_auth()
    # A limit per client address, and one per user name that a client changing
    # its address cannot step around (app/services/login_limiter.py).
    keys = {
        "addr:" + (request.client.host if request.client else "unknown"): settings.analytics_login_max_failures,
        _username_key(body.username): settings.analytics_login_max_failures_per_username,
    }
    # The configured user's lockout is never evicted from the bounded table.
    pinned = {_username_key(settings.analytics_username)} if settings.analytics_username else set()
    wait = _login_limiter.attempt(keys, window_seconds=settings.analytics_login_window_seconds, pinned=pinned)
    if wait is not None:
        seconds = max(1, math.ceil(wait))
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed logins. Try again in {seconds} seconds.",
            headers={"Retry-After": str(seconds)},
        )
    # compare_digest keeps the comparison constant-time so a caller cannot infer
    # the credentials one character at a time from response timing.
    # Compared as UTF-8 bytes: compare_digest raises TypeError on a str with
    # any non-ASCII character, so "opérateur" answered 500 instead of 401.
    username_ok = secrets.compare_digest(body.username.encode(), settings.analytics_username.encode())
    password_ok = secrets.compare_digest(body.password.encode(), settings.analytics_password.encode())
    if username_ok and password_ok:
        _login_limiter.succeeded(keys)
        return LoginResponse(token=_create_token(body.username))
    raise HTTPException(status_code=401, detail="Invalid credentials.")


@router.get("/analytics/data")
def analytics_data(
    # Bounded: a cutoff further back than year 1 overflowed datetime (500).
    days: int = Query(30, ge=0, le=36500),
    _user: str = Depends(_verify_token),
):
    return get_analytics_data(days)
