"""This module initializes the application."""

from fastapi import FastAPI

from rag_youtube_transcripts.app.router import router


app: FastAPI = FastAPI(title="YouTube Search API")
app.include_router(router)
