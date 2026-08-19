"""This module contains functionality for fetching YouTube video transcripts and their
corresponding metadata, that is, video ID, creation date, and title, from the YouTube
Data API.
"""

import os
from datetime import datetime
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from httpx import Client, Response
from omegaconf import DictConfig
from youtube_transcript_api import YouTubeTranscriptApi

from rag_youtube_transcripts.config import Config
from rag_youtube_transcripts.logger import logger


load_dotenv(Config.Paths.env)

PARAMS: DictConfig = Config.load_params(Path(__file__).stem)


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
        max_transcripts (int, optional): Maximum number of video transcripts to fetch.
        Defaults to PARAMS.max_transcripts.

    Returns:
        pl.DataFrame: YouTube video transcripts and corresponding metadata, that is,
        video ID, creation date, and title.
    """
    try:
        video_ids: set[str] = set(pl.read_parquet(Config.Paths.transcripts).get_column("video_id"))
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
