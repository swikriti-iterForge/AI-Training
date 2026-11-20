"""Day 12 - password hashing, JWT tokens, and the 'who is logged in' dependency."""

from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from pwdlib.hashers.bcrypt import BcryptHasher

from bookverse.core.config import get_settings
from bookverse.db import SessionDep
from bookverse.models import User

password_hash = PasswordHash((BcryptHasher(),))
# Only EXTRACTS the token from the "Authorization: Bearer ..." header (and powers the
# Authorize button in /docs). Checking it is done in get_current_user below.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def hash_password(password: str) -> str:
    return password_hash.hash(password)  # only the hash is ever stored


def verify_password(password: str, hashed: str) -> bool:
    return password_hash.verify(password, hashed)


def create_access_token(user_id: int, minutes: int | None = None) -> str:
    """The token holds only the user id and an expiry. A JWT is signed, NOT encrypted:
    anyone can read its payload, so never put secrets in it."""
    s = get_settings()
    now = datetime.now(UTC)
    payload = {"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=minutes or s.access_token_expire_minutes)}
    return jwt.encode(payload, s.jwt_secret_key, algorithm=s.jwt_algorithm)


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], session: SessionDep) -> User:
    s = get_settings()
    try:
        user_id = int(jwt.decode(token, s.jwt_secret_key, algorithms=[s.jwt_algorithm])["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
