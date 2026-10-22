"""Analytics endpoints with JWT authentication."""

import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import settings
from app.schemas.analytics import LoginRequest, LoginResponse
from app.services.analytics_service import get_analytics_data

router = APIRouter(tags=["analytics"])
security = HTTPBearer()

ALGORITHM = "HS256"


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
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token.")
        return username
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token.") from exc


@router.post("/analytics/login")
async def analytics_login(body: LoginRequest):
    _require_configured_auth()
    # compare_digest keeps the comparison constant-time so a caller cannot infer
    # the credentials one character at a time from response timing.
    username_ok = secrets.compare_digest(body.username, settings.analytics_username)
    password_ok = secrets.compare_digest(body.password, settings.analytics_password)
    if username_ok and password_ok:
        return LoginResponse(token=_create_token(body.username))
    raise HTTPException(status_code=401, detail="Invalid credentials.")


@router.get("/analytics/data")
async def analytics_data(days: int = 30, _user: str = Depends(_verify_token)):
    return get_analytics_data(days)
