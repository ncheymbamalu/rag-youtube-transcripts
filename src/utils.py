"""This module contains objects and functions that are used in other modules and scripts."""

import json
import os
import re
import string
import textwrap

from datetime import datetime

import numpy as np
import polars as pl
import torch

from chonkie import TokenChunker
from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletion
from httpx import Client, Response
from nltk.corpus import stopwords
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi

from src.config import Config
from src.logger import logger

load_dotenv(Config.env)

groq_client: Groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

embedding_model: SentenceTransformer = SentenceTransformer(
    model_name_or_path=Config.load_params("embedding_model"),
    trust_remote_code=True
)

reranker_model: CrossEncoder = CrossEncoder(
    model_name_or_path=Config.load_params("reranker_model"),
    activation_fn=torch.nn.Sigmoid()
)

knowledge_base: pl.LazyFrame = (
    pl.scan_parquet(Config.embeddings)
    .join(
        other=pl.scan_parquet(Config.transcripts),
        how="inner",
        on="video_id"
    )
    .sort(by=["video_id", "chunk_index"])
    .select(
        "video_id",
        "title",
        "chunk",
        pl.col("embedding").str.json_decode()
    )
)


@logger.catch
def fetch_transcripts(
    youtube_channel_id: str,
    max_results: int = Config.load_params("max_results"),
) -> pl.DataFrame:
    """Fetches YouTube video transcripts and corresponding metadata from the YouTube
    Data GET endpoint and returns a pl.DataFrame.

    Args:
        youtube_channel_id (str): ID of the YouTube channel whose video transcripts
        will be fetched.
        max_results (int, optional): Maximum number of video transcripts to fetch.
        Defaults to Config.load_params("max_results").

    Returns:
        pl.DataFrame: YouTube video transcripts and corresponding metadata, that is,
        video ID, creation date, and title.
    """
    try:
        video_ids: list[str] = pl.read_parquet(Config.transcripts)["video_id"].to_list()
        params: dict[str, int | list[str] | str] = {
            "key": os.getenv("YOUTUBE_DATA_API_KEY"),
            "channelId": youtube_channel_id,
            "part": ["snippet", "id"],
            "order": "date",
            "maxResults": max_results
        }
        schema: list[str] = ["video_id", "creation_date", "title", "transcript"]
        with Client() as client:
            response: Response = client.get(url=Config.youtube_data_api, params=params)
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
                        logger.info(f"Skipping `{title}`. Its transcript has already been fetched.")
                    else:
                        try:
                            transcript = " ".join(
                                snippet.text.strip().lower()
                                for snippet in YouTubeTranscriptApi().fetch(video_id)
                            )
                            logger.info(f"SUCCESS: The transcript for `{title}` has been fetched.")
                        except Exception:
                            transcript = "skip"
                            logger.info(f"Skipping `{title}`. Its transcript is unavailable.")
                    record += [transcript]
                    records.append(dict(zip(schema, record)))
                data: pl.DataFrame = (
                    pl.DataFrame(records)
                    .with_columns(
                        pl.col(col_name)
                        .str.replace_all(r"\s{2,}", " ")
                        .str.replace_many(
                            ["&#39;", "&quot;", "&amp;"],
                            ["'", "'", "&"]
                        )
                        for col_name in ["title", "transcript"]
                    )
                    .filter(pl.col("transcript").ne("skip"))
                )
                return data
            logger.info(
                f"Invalid request. Unable to access videos from the YouTube channel ID, \
{youtube_channel_id}"
            )
            return pl.DataFrame(schema=schema)
    except Exception as e:
        raise e


def add_context_to_chunk(
    transcript: str,
    chunk: str,
    llm: str = Config.load_params("llm").get("contextual_chunking"),
    temperature: float = Config.load_params("temperature").get("contextual_chunking"),
    max_output_tokens: int = Config.load_params("max_output_tokens").get("contextual_chunking"),
) -> str:
    """Adds context to a YouTube video transcript's chunk.

    Args:
        transcript (str): YouTube video transcript.
        chunk (str): Subset of the YouTube video transcript.
        llm (str, optional): Model used to generate the chunk's context.
        Defaults to Config.load_params("llm").get("contextual_chunking").
        temperature (float, optional): Parameter between 0 and 2, inclusive, that contols the
        randomness of the llm's output. The lower the temperature, the more repeatable the
        response. Defaults to Config.load_params("temperature").get("contextual_chunking").
        max_output_tokens (int, optional): Maximum number of tokens used to generate the llm's
        output. Defaults to Config.load_params("max_output_tokens").get("contextual_chunking").

    Returns:
        str: Chunk that's prefixed with context.
    """
    try:
        system_prompt: str = "You are an expert in document analysis. Your task is to provide \
brief, relevant context for a chunk of text."
        user_prompt: str = f"""\
Here is the document:
<document>
{transcript}
</document>

Here is the chunk of text:
<chunk>
{chunk}
</chunk>

Provide a concise context (2-4 sentences) for this chunk, considering the following guidelines:
1. Identify the main topic or concept discussed in the chunk.
2. Mention any relevant information or comparisons from the broader document context.
3. If applicable, note how the information contained in the chunk relates to the overall theme or \
purpose of the document.

Please provide a short, succinct context to situate this chunk within the overall document for the \
purposes of improving search retrieval of the chunk.
Respond only with the context.\
        """
        completion: ChatCompletion = groq_client.chat.completions.create(
            model=llm,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_output_tokens,
        )
        context: str = completion.choices[0].message.content.strip().lower()
        return f"{context}\n{chunk}"
    except Exception as e:
        raise e


def encode_transcripts(
    data: pl.DataFrame,
    model: SentenceTransformer = embedding_model,
    chunk_size: int = Config.load_params("chunk_size"),
) -> pl.DataFrame:
    """Splits each YouTube video transcript into smaller contextualized chunks and
    generates embeddings for each.

    Args:
        data (pl.DataFrame): YouTube video transcripts and corresponding metadata,
        that is, video ID, creation date, and title.
        model (SentenceTransformer): Bi-encoder text embedding model that's used to
        convert a string of text to embeddings (1-D array of floating point numbers).
        Defaults to embedding_model.
        chunk_size (int, optional): Maximum number of tokens per chunk.
        Defaults to Config.load_params("chunk_size")

    Returns:
        pl.DataFrame: YouTube video transcripts' embeddings and their corresponding video ID.
    """
    try:
        chunker: TokenChunker = TokenChunker(
            tokenizer=model.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=(chunk_size // 10)
        )
        dfs: list[pl.DataFrame] = []
        for idx, transcript in enumerate(tqdm(
            iterable=data["transcript"],
            unit="transcript",
            desc="Splitting the transcripts into chunks and generating their embeddings"
        )):
            chunks: list[str] = [
                add_context_to_chunk(transcript, chunk.text.strip())
                for chunk in chunker(transcript)
            ]
            embeddings: list[str] = [
                json.dumps(
                    model
                    .encode(f"search_document: {chunk}", normalize_embeddings=True)
                    .tolist()
                )
                for chunk in chunks
            ]
            video_ids: list[str] = [data[idx, "video_id"]] * len(chunks)
            records: list[dict[str, int | str]] = [
                {
                    "video_id": video_id,
                    "chunk_index": idx + 1,
                    "chunk": chunk,
                    "embedding": emb
                }
                for idx, (video_id, chunk, emb) in enumerate(zip(video_ids, chunks, embeddings))
            ]
            dfs.append(pl.DataFrame(records))
        data = (
            pl.concat(dfs, how="vertical")
            .cast({"chunk_index": pl.Int16})
            .sort(by=["video_id", "chunk_index"])
        )
        return data
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
        return query.strip().lower()
    except Exception as e:
        raise e


def get_semantic_search_results(
    query: str,
    k: int = Config.load_params("k"),
    threshold: float = Config.load_params("threshold"),
) -> pl.DataFrame:
    """Returns a pl.DataFrame that contains the title and URL of the top k
    YouTube videos whose chunk has the highest degree of semantic similarity
    with the input query.

    Args:
        query (str): Input query
        k (int): Number of results to return. Defaults to Config.load_params("k").
        threshold (float, optional): Threshold probability used to filter out less
        relevant results. Defaults to Config.load_params("threshold").

    Returns:
        pl.DataFrame: Title and URL of the top k YouTube videos whose chunk has the
        strongest contextual relationship with the input query. 
    """
    try:
        query: str = preprocess_query(query)
        keywords: str = "|".join(
            word for word in query.split() if word not in stopwords.words("english")
        )
        filtered_knowledge_base: pl.LazyFrame = (
            knowledge_base
            .filter(
                pl.concat_str((pl.col("title").str.to_lowercase(), "chunk"), separator=": ")
                .str.contains(keywords)
            )
        )
        if filtered_knowledge_base.collect().is_empty():
            return pl.DataFrame({"title": None, "url": None}).cast(pl.String)
        return (
            filtered_knowledge_base
            .with_columns(
                pl.Series(
                    name="cosine_similarity",
                    values=(
                        np.array(filtered_knowledge_base.collect()["embedding"].to_list())
                        .dot(
                            embedding_model
                            .encode(f"search_query: {query}", normalize_embeddings=True)
                            .reshape(-1, 1)
                        )
                        .ravel()
                    )
                )
            )
            .filter(pl.col("cosine_similarity").gt(0))
            .sort(by="cosine_similarity", descending=True)
            .head(100)
            .with_columns(
                pl.concat_str((pl.col("title").str.to_lowercase(), "chunk"), separator=": ")
                .map_elements(
                    lambda chunk: reranker_model.predict((query, chunk)),
                    return_dtype=pl.Float64
                )
                .alias("relevance_score")
            )
            .filter(pl.col("relevance_score").ge(threshold))
            .sort(by="relevance_score", descending=True)
            .unique(subset=["title", "video_id"], maintain_order=True)
            .select(
                "title",
                pl.concat_str((pl.lit("https://youtu.be/"), pl.col("video_id"))).alias("url")
            )
            .head(k)
            .collect()
        )
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
