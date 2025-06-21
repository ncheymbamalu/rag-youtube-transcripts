"""This module initializes the application."""

from fastapi import FastAPI

from src.app.router import router as main_router

app: FastAPI = FastAPI(title="YouTube Search API")
app.include_router(main_router)
