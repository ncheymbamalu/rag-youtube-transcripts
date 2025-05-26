"""This module provides functionality for the YouTube video transcripts ETL pipeline."""

import time

from functools import reduce
from multiprocessing import Pool

import polars as pl

from src.config import Config
from src.logger import logger
from src.utils import encode_transcripts, fetch_transcripts


@logger.catch
def main() -> None:
    """Fetches video transcripts and corresponding metadata from a list of YouTube channel IDs,
    generates their embeddings, and writes the resulting data to ./artifacts/data.
    """
    try:
        # a list of YouTube video IDs whose transcripts have already been fetched
        video_ids: list[str] = pl.read_parquet(Config.transcripts)["video_id"].to_list()

        # fetch the YouTube video transcripts
        with Pool() as pool:
            data: pl.DataFrame = (
                reduce(
                    lambda left, right: pl.concat((left, right), how="vertical"),
                    pool.imap(fetch_transcripts, Config.load_params("youtube_channel_ids"))
                )
                .filter(~pl.col("video_id").is_in(video_ids))
            )
        if data.is_empty():
            logger.info("There are no new transcripts. Skipping the embedding process.")
        else:
            # update ./artifacts/data/transcripts.parquet
            (
                pl.concat(
                    (
                        data,
                        pl.read_parquet(Config.transcripts)
                    ),
                    how="vertical"
                )
                .sort(by=["creation_date", "video_id"], descending=[True, False])
                .write_parquet(Config.transcripts)
            )

            # update ./artifacts/data/embeddings.parquet
            logger.info("Starting the embedding process...")
            start: float = time.perf_counter()
            (
                pl.concat(
                    (
                        data.pipe(encode_transcripts),
                        pl.read_parquet(Config.embeddings)
                    ),
                    how="vertical"
                )
                .sort(by="video_id")
                .write_parquet(Config.embeddings)
            )
            logger.info(
                f"Finished! It took ~{((time.perf_counter() - start)/60):.2f} minutes to generate \
embeddings for {data.shape[0]:_} YouTube video transcripts."
            )
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
