"""This module contains functionality for retrieving the YouTube video transcript chunks
that are most relevant to an input query, via a cascade of BM25 filtering, cosine
similarity, and cross-encoder re-ranking.
"""

import re
import string
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from polars.datatypes.classes import DataTypeClass

from rag_youtube_transcripts.config import Config
from rag_youtube_transcripts.models import (
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE,
    RERANKER_MODEL,
    SPELL_CHECKER,
)


PARAMS: DictConfig = Config.load_params(Path(__file__).stem)


def preprocess_query(query: str) -> str:
    """Pre-processes the input query.

    Args:
        query (str): Input query.

    Returns:
        str: Pre-processed input query.
    """
    try:
        query = re.sub(f"[{string.punctuation}]", " ", query)
        query = re.sub(r"\s{2,}", " ", query)
        query = " ".join(
            SPELL_CHECKER.correction(word) if SPELL_CHECKER.candidates(word) else word
            for word in query.strip().lower().split()
        )
        return query
    except Exception as e:
        raise e


def get_semantic_search_results(
    query: str,
    k: int = PARAMS.k,
    threshold: float = PARAMS.relevance_threshold,
) -> pl.DataFrame:
    """Returns a pl.DataFrame that contains the title and URL of the top k
    YouTube videos whose chunk has the highest degree of semantic similarity
    with the input query.

    Args:
        query (str): Input query
        k (int, optional): Number of results to return. Defaults to PARAMS.k.
        threshold (float, optional): Threshold probability used to filter out less
        relevant results. Defaults to PARAMS.relevance_threshold.

    Returns:
        pl.DataFrame: Title, URL, start, and end marks of the top k YouTube videos
        whose chunk has the strongest contextual relationship with the input query. 
    """
    try:
        # pre-process the input query
        query = preprocess_query(query)

        # create the empty result
        schema: dict[str, DataTypeClass] = {
            "title": pl.String,
            "url": pl.String,
            "start": pl.Float64,
            "end": pl.Float64,
            "excerpt": pl.String
        }
        empty_result: pl.DataFrame = pl.DataFrame(schema=schema)

        # BM25
        candidates: pl.DataFrame = (
            pl.scan_parquet(Config.Paths.bm25_data)
            .filter(pl.col("token").is_in(query.split()))
            .group_by("video_id", "chunk_index")
            .agg(pl.sum("score"))
            .join(
                KNOWLEDGE_BASE,
                how="inner",
                on=["video_id", "chunk_index"],
                maintain_order="left"
            )
            .drop("chunk_index", "score")
            .collect()
        )
        if candidates.is_empty():
            return empty_result

        # cosine similarity
        query_embedding: np.ndarray = EMBEDDING_MODEL.encode(
            f"search_query: {query}",
            normalize_embeddings=True
        )
        cos_sims: np.ndarray = candidates.get_column("embedding").to_numpy().dot(query_embedding)
        candidates = (
            candidates
            .with_columns(pl.Series("cosine_similarity", cos_sims, dtype=pl.Float32))
            .filter(pl.col("cosine_similarity").gt(0))
            .sort("cosine_similarity", descending=True)
            .drop("embedding", "cosine_similarity")
            .head(250)
            .with_columns(
                pl.concat_str(pl.col("title").str.to_lowercase(), "chunk", separator=": ")
                .alias("titled_chunk")
            )
        )
        if candidates.is_empty():
            return empty_result

        # cross-encoder re-ranking
        rel_scores: np.ndarray = RERANKER_MODEL.predict(
            [(query, chunk) for chunk in candidates.get_column("titled_chunk")],
            batch_size=32,
            show_progress_bar=False,
        )
        candidates = (
            candidates
            .with_columns(pl.Series("relevance_score", rel_scores, dtype=pl.Float32))
            .filter(pl.col("relevance_score").ge(threshold))
            .sort("relevance_score", descending=True)
            .select(
                "title",
                pl.concat_str(pl.lit("https://youtu.be/"), "video_id").alias("url"),
                "start",
                "end",
                pl.col("chunk").alias("excerpt")
            )
            .limit(k)
        )
        return candidates
    except Exception as e:
        raise e
