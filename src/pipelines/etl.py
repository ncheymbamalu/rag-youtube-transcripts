"""This module provides functionality for the YouTube video transcripts ETL pipeline."""

import time

import polars as pl

from joblib import Parallel, delayed
from tqdm import tqdm

from src.config import Config
from src.logger import logger
from src.utils import encode_transcripts, fetch_transcripts


@logger.catch
def main() -> None:
    """Fetches video transcripts and corresponding metadata from a list of YouTube channel IDs,
    generates their embeddings, and writes the resulting data to ./artifacts/data/.
    """
    try:
        # fetch the YouTube video transcripts
        dfs: list[pl.DataFrame] = Parallel(n_jobs=-1)(
            delayed(fetch_transcripts)(youtube_channel_id)
            for youtube_channel_id in tqdm(
                iterable=Config.load_params("youtube_channel_ids"),
                unit="YouTube Channel ID",
                desc="Fetching YouTube video transcripts"
            )
        )
        data: pl.DataFrame = pl.concat(dfs, how="vertical")
        if data.is_empty():
            logger.info("There are no new transcripts. Skipping the embedding process.")
        else:
            # update ./artifacts/data/transcripts.parquet
            (
                pl.concat((data, pl.read_parquet(Config.transcripts)), how="vertical")
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
                .sort(by=["video_id", "chunk_index"])
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
