"""This module contains utility functions that are used by other modules."""

import json
import os

from datetime import datetime

import polars as pl

from chonkie import TokenChunker
from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletion
from httpx import Client, Response
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi

from src.config import Config
from src.logger import logger

load_dotenv(Config.env)

groq_client: Groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

embedding_model: SentenceTransformer = SentenceTransformer(
    model_name_or_path=Config.load_params("embedding_model"),
    trust_remote_code=True
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
                    title: str = item.get("snippet").get("title")
                    record: list[datetime | str] = [
                        video_id,
                        datetime.strptime(
                            item.get("snippet").get("publishedAt"),
                            "%Y-%m-%dT%H:%M:%S%z"
                        ),
                        title
                    ]
                    try:
                        transcript: str = " ".join(
                            snippet.text.strip().lower()
                            for snippet in YouTubeTranscriptApi().fetch(video_id)
                        )
                        records.append(dict(zip(schema, record + [transcript])))
                        logger.info(f"SUCCESS: The transcript for `{title}` has been fetched.")
                    except Exception:
                        logger.info(
                            f"ERROR: The transcript is unavailable. `{title}` will be removed."
                        )
                        records.append(dict(zip(schema, record + ["unavailable"])))
                data: pl.DataFrame = (
                    pl.DataFrame(records)
                    .filter(pl.col("transcript").ne(pl.lit("unavailable")))
                    .with_columns(
                        pl.col(col_name)
                        .str.replace_many(
                            ["&#39;", "&quot;", "&amp;", "  ", "\n"],
                            ["'", "'", "&", " ", " "]
                        )
                        for col_name in ["title", "transcript"]
                    )
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
        for idx, transcript in enumerate(data["transcript"]):
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
