"""This module contains objects and functions that are used in other modules and scripts."""

import os
import re
import string
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from chonkie import TokenChunker
from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletion
from groq.types.chat.chat_completion import Choice as ChatCompletionChoice
from httpx import Client, Response
from nltk.corpus import stopwords
from omegaconf import DictConfig
from polars.datatypes.classes import DataTypeClass
from sentence_transformers import CrossEncoder, SentenceTransformer
from spellchecker import SpellChecker
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, logging
from youtube_transcript_api import YouTubeTranscriptApi

from rag_youtube_transcripts.config import Config
from rag_youtube_transcripts.logger import logger


load_dotenv(Config.Paths.env)

logging.set_verbosity_error()

PARAMS: DictConfig = Config.load_params(Path(__file__).stem)
GROQ_CLIENT: Groq = Groq(api_key=os.getenv("GROQ_API_KEY", ""), max_retries=5, timeout=30.0)
EMBEDDING_MODEL: SentenceTransformer = SentenceTransformer(
    model_name_or_path=PARAMS.models.embedding,
    trust_remote_code=True
)
RERANKER_MODEL: CrossEncoder = CrossEncoder(
    model_name_or_path=PARAMS.models.reranking,
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


@logger.catch
def fetch_transcripts(
    youtube_channel_id: str,
    max_transcripts: int = PARAMS.max_transcripts,
) -> pl.DataFrame:
    """Fetches YouTube video transcripts and corresponding metadata from the YouTube
    Data GET endpoint and returns a pl.DataFrame.

    Args:
        youtube_channel_id (str): ID of the YouTube channel whose video transcripts
        will be fetched.
        max_results (int, optional): Maximum number of video transcripts to fetch.
        Defaults to PARAMS.max_transcripts.

    Returns:
        pl.DataFrame: YouTube video transcripts and corresponding metadata, that is,
        video ID, creation date, and title.
    """
    try:
        video_ids: list[str] = (
            pl.read_parquet(Config.Paths.transcripts)
            .get_column("video_id")
            .to_list()
        )
        params: dict[str, int | list[str] | str] = {
            "key": os.getenv("YOUTUBE_DATA_API_KEY", ""),
            "channelId": youtube_channel_id,
            "part": ["snippet", "id"],
            "order": "date",
            "maxResults": max_transcripts
        }
        schema: pl.Schema = pl.Schema({
            "video_id": pl.String,
            "creation_date": pl.Datetime(time_unit="us", time_zone="UTC"),
            "title": pl.String,
            "transcript": pl.String
        })
        with Client() as client:
            response: Response = client.get(PARAMS.youtube_data_api, params=params)
            if response.status_code == 200:
                records: list[dict[str, datetime | str]] = []
                for item in response.json().get("items"):
                    video_id: str = item.get("id").get("videoId")
                    creation_date: str = item.get("snippet").get("publishedAt")
                    title: str = item.get("snippet").get("title")
                    record: list[datetime | str] = [
                        video_id,
                        datetime.strptime(creation_date, "%Y-%m-%dT%H:%M:%S%z"),
                        title,
                    ]
                    if video_id in video_ids:
                        transcript: str = "skip"
                        logger.info(f"Skipping <green>{title}</>. Transcript already fetched.")
                    else:
                        try:
                            transcript = " ".join(
                                snippet.text.strip().lower()
                                for snippet in YouTubeTranscriptApi().fetch(video_id)
                            )
                            logger.info(
                                f"SUCCESS: The transcript for <green>{title}</> has been fetched."
                            )
                        except Exception:
                            transcript = "skip"
                            logger.info(f"Skipping <green>{title}</>. Transcript is unavailable.")
                    record.append(transcript)
                    records.append(dict(zip(schema.names(), record, strict=True)))
                data: pl.DataFrame = (
                    pl.DataFrame(records)
                    .with_columns(
                        pl.col(col)
                        .str.replace_all(r"\s{2,}", " ")
                        .str.replace_many(
                            ["&#39;", "&quot;", "&amp;"],
                            ["'", "'", "&"]
                        )
                        for col in ("title", "transcript")
                    )
                    .filter(pl.col("transcript").ne("skip"))
                )
                return data
            logger.info(
                "Invalid request. Unable to access videos from the YouTube channel ID, "
                f"<green>{youtube_channel_id}</>"
            )
            return pl.DataFrame(schema=schema)
    except Exception as e:
        raise e


def add_context_to_chunk(
    transcript: str,
    chunk: str,
    llm: str = PARAMS.contextual_chunking.llm,
    temperature: float = PARAMS.contextual_chunking.temperature,
    max_output_tokens: int = PARAMS.contextual_chunking.max_output_tokens,
) -> str:
    """Adds context to a YouTube video transcript's chunk.

    Args:
        transcript (str): YouTube video transcript.
        chunk (str): Subset of the YouTube video transcript.
        llm (str, optional): Model used to generate the chunk's context.
        Defaults to PARAMS.contextual_chunking.llm.
        temperature (float, optional): Parameter between 0 and 2, inclusive, that contols the
        randomness of the llm's output. The lower the temperature, the more repeatable the
        response. Defaults to PARAMS.contextual_chunking.temperature.
        max_output_tokens (int, optional): Maximum number of tokens used to generate the llm's
        output. Defaults to PARAMS.contextual_chunking.max_output_tokens.

    Returns:
        str: Chunk that's prefixed with context.
    """
    try:
        system_prompt: str = PARAMS.contextual_chunking.system_prompt
        user_prompt: str = PARAMS.contextual_chunking.user_prompt
        reasoning_effort: str = PARAMS.contextual_chunking.reasoning_effort
        completion: ChatCompletion = GROQ_CLIENT.chat.completions.create(
            model=llm,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt.format(transcript=transcript, chunk=chunk)
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort
        )
        choice: ChatCompletionChoice = completion.choices[0]
        if choice.finish_reason == "length" or not choice.message.content:
            logger.warning(
                f"Empty/truncated context (finish_reason={choice.finish_reason}). "
                f"Using original chunk: {chunk[:80]}..."
            )
            return chunk
        context: str = choice.message.content.strip().lower()
        return f"{context}\n{chunk}"
    except Exception as e:
        raise e


def encode_transcripts(data: pl.DataFrame) -> pl.DataFrame:
    """Splits each YouTube video transcript into smaller contextualized chunks and
    generates embeddings for each.

    Args:
        data (pl.DataFrame): YouTube video transcripts and corresponding metadata,
        that is, video ID, creation date, and title.

    Returns:
        pl.DataFrame: YouTube video transcripts' embeddings and their corresponding video ID.
    """
    try:
        tokenizer: PreTrainedTokenizerFast = EMBEDDING_MODEL.tokenizer
        max_transcript_tokens: int = PARAMS.encoding.max_transcript_tokens
        chunk_size: int = PARAMS.encoding.chunk_size
        chunker: TokenChunker = TokenChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=(chunk_size // 10)
        )
        dfs: list[pl.DataFrame] = []
        for video_id, transcript in enumerate(tqdm(
            iterable=zip(data.get_column("video_id"), data.get_column("transcript"), strict=True),
            desc="Splitting transcripts into contextual chunks and generating embeddings for each",
            total=data.height,
            unit="transcript",
            
        )):
            tokens: list[str] = tokenizer.tokenize(transcript)
            if len(tokens) > max_transcript_tokens:
                tokens = tokens[:max_transcript_tokens]
                token_ids: list[int] = tokenizer.convert_tokens_to_ids(tokens)
                transcript = tokenizer.decode(token_ids).strip()
            chunks: list[str] = [
                add_context_to_chunk(transcript, chunk.text.strip())
                for chunk in chunker(transcript)
            ]
            if not chunks:
                logger.warning(f"No chunks produced for <green>{video_id}</>. Skipping.")
                continue
            embedding_dim: int = EMBEDDING_MODEL.get_sentence_embedding_dimension()
            embeddings: np.ndarray = EMBEDDING_MODEL.encode(  # (len(chunks), embedding_dim)
                [f"search_document: {chunk}" for chunk in chunks],
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            )
            video_ids: list[str] = [video_id] * len(chunks)
            chunk_indices: range = range(1, len(chunks) + 1)
            records: pl.DataFrame = pl.DataFrame({
                "video_id": video_ids,
                "chunk_index": pl.Series(values=chunk_indices, dtype=pl.Int16),
                "chunk": chunks,
                "embedding": pl.Series(values=embeddings, dtype=pl.Array(pl.Float32, embedding_dim))
            })
            dfs.append(records)
        data = pl.concat(dfs, how="vertical").sort("video_id", "chunk_index")
        return data
    except Exception as e:
        raise e


def create_bm25_dataset(k1: float = 1.5, b: float = 0.75) -> None:
    """Creates a dataset that stores each token's BM25 score with repect to
    each contextualized chunk and writes the results to `./artifacts/data/bm25.parquet.`
    """
    try:
        stop_words: frozenset = frozenset([
            word.translate(str.maketrans("", "", string.punctuation))
            for word in stopwords.words("english")
        ])

        # create the base plan
        # NOTE: each unique (video_id, chunk_index) pair represents a unique chunk
        plan: pl.LazyFrame = (
            pl.scan_parquet(Config.Paths.embeddings)
            .select(
                "video_id",
                "chunk_index",
                (
                    pl.col("chunk")
                    .str.to_lowercase()
                    .str.replace_all(r"[^a-z0-9\s]", "")
                    .str.replace_all(r"\s+", " ")
                    .str.strip_chars()
                    .str.split(" ")
                    .list.eval(pl.element().filter(~pl.element().is_in(stop_words)))
                    .alias("tokens")
                )
            )
            .with_columns(pl.col("tokens").list.len().alias("n_tokens"))
        )

        # get the global stats, that is, average token count and the total number of chunks
        stats: pl.DataFrame = (
            plan
            .select(
                pl.mean("n_tokens").alias("avg_n_tokens"),
                pl.len().alias("n_chunks")
            )
            .collect()
        )
        avg_n_tokens: float = stats.get_column("avg_n_tokens").item()
        n_chunks: int = stats.get_column("n_chunks").item()

        # create the term frequency plan
        tf_plan: pl.LazyFrame = (
            plan
            .explode("tokens")
            .rename({"tokens": "token"})
            .filter(pl.col("token").ne(""))  # removes empty strings
            .group_by("video_id", "chunk_index", "n_tokens", "token", maintain_order=True)
            .agg(pl.len().alias("tf"))
        )

        # create the inverse document frequency plan
        idf_plan: pl.LazyFrame = (
            tf_plan
            .group_by("token")
            .agg(pl.len().alias("n"))
            .with_columns(
                ((n_chunks - pl.col("n") + 0.5) / (pl.col("n") + 0.5) + 1).log().alias("idf")
            )
        )

        # calculate the BM25 score for each unique (video_id, chunk_index, token) group and ...
        # write to `./artifacts/data/bm25.parquet`
        (
            tf_plan
            .join(idf_plan, on="token", maintain_order="left")
            .with_columns(
                (
                    pl.col("idf")
                    *
                    (
                        (pl.col("tf") * (k1 + 1))
                        /
                        (pl.col("tf") + k1 * (1 - b + b * (pl.col("n_tokens") / avg_n_tokens)))
                    )
                )
                .alias("score")
            )
            .select("video_id", "chunk_index", "token", "score")
            .sink_parquet(Config.Paths.bm25_data)
        )
    except Exception as e:
        raise e


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
    k: int = PARAMS.semantic_search.k,
    threshold: float = PARAMS.semantic_search.relevance_threshold,
) -> pl.DataFrame:
    """Returns a pl.DataFrame that contains the title and URL of the top k
    YouTube videos whose chunk has the highest degree of semantic similarity
    with the input query.

    Args:
        query (str): Input query
        k (int, optional): Number of results to return. Defaults to PARAMS.semantic_search.k.
        threshold (float, optional): Threshold probability used to filter out less
        relevant results. Defaults to PARAMS.semantic_search.relevance_threshold.

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


def wrap_text(text: str) -> str:
    """Processes a string of text so that it's more readable when printed out.

    Args:
        text (str): String of text.

    Returns:
        str: Processed string of text.
    """
    try:
        return textwrap.fill(text, width=100)
    except Exception as e:
        raise e
