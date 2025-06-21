"""This module contains the application's endpoints."""

from fastapi import APIRouter

from src.rag import generate_response

router: APIRouter = APIRouter()


@router.get("/healthz", response_model=dict[str, str], summary="Check API Health")
def check_api_health():
    return {"status": "ok"}


@router.get("/search", response_model=dict[str, str], summary="Retrieval Augmented Generation")
def get_response(query: str):
    response: str = generate_response(query)
    return {"response": response}
