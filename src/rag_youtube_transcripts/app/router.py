"""This module contains the application's router."""

from fastapi import APIRouter

from rag_youtube_transcripts.app import endpoint


router: APIRouter = APIRouter()
router.include_router(endpoint.router, prefix="/rag-youtube")
