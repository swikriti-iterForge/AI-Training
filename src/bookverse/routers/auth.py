"""Day 12 - register and login."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from bookverse.core.security import CurrentUser, create_access_token, hash_password, verify_password
from bookverse.db import SessionDep
from bookverse.models import User
from bookverse.schemas import Token, UserCreate, UserRead

router = APIRouter(tags=["auth"])


@router.post("/register", response_model=UserRead, status_code=201)
async def register(data: UserCreate, session: SessionDep):
    user = User(email=data.email.lower(), hashed_password=hash_password(data.password))
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "An account with this email already exists")
    return user


@router.post("/login", response_model=Token)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], session: SessionDep):
    """OAuth2 form login. The standard calls the field `username`; here it is the email."""
    user = await session.scalar(select(User).where(User.email == form.username.lower()))
    # Same message whether the email or the password is wrong: don't reveal which emails exist.
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(401, "Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    return Token(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserRead)
async def me(user: CurrentUser):
    return user
