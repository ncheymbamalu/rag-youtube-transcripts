"""This module instantiates the project's shared, expensive-to-construct objects, that is,
the embedding and re-ranking models, the Groq client, the spell checker, and the knowledge
base, so that each is created once per process and imported wherever it's needed.
"""

import os
from pathlib import Path

import polars as pl
import torch
from dotenv import load_dotenv
from groq import Groq
from omegaconf import DictConfig
from sentence_transformers import CrossEncoder, SentenceTransformer
from spellchecker import SpellChecker
from transformers import logging

from rag_youtube_transcripts.config import Config


load_dotenv(Config.Paths.env)

logging.set_verbosity_error()

PARAMS: DictConfig = Config.load_params(Path(__file__).stem)
GROQ_CLIENT: Groq = Groq(api_key=os.getenv("GROQ_API_KEY", ""), max_retries=5, timeout=30.0)
EMBEDDING_MODEL: SentenceTransformer = SentenceTransformer(
    model_name_or_path=PARAMS.embedding,
    trust_remote_code=True
)
RERANKER_MODEL: CrossEncoder = CrossEncoder(
    model_name_or_path=PARAMS.reranker,
    activation_fn=torch.nn.Sigmoid()
)
KNOWLEDGE_BASE: pl.LazyFrame = (
    pl.scan_parquet(Config.Paths.embeddings)
    .with_columns(pl.col("chunk_index").max().over("video_id").alias("chunk_count"))
    .join(
        pl.scan_parquet(Config.Paths.transcripts).select("video_id", "title"),
        how="inner",
        on="video_id"
    )
    .with_columns(
        ((pl.col("chunk_index") - 1) / pl.col("chunk_count")).round(2).alias("start"),
        (pl.col("chunk_index") / pl.col("chunk_count")).round(2).alias("end")
    )
    .select(
        "video_id",
        "chunk_index",
        "title",
        "start",
        "end",
        "chunk",
        "embedding"
    )
    .sort("video_id", "start")
)
SPELL_CHECKER: SpellChecker = SpellChecker(distance=1)
