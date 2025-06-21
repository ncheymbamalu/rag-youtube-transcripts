"""This module contains the application's router."""

from fastapi import APIRouter

import src.app.endpoint as endpoint

router: APIRouter = APIRouter()


router.include_router(endpoint.router, prefix="/rag-youtube")
