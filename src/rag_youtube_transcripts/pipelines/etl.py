"""This script executes the YouTube video transcripts ETL pipeline."""

import time

import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

from rag_youtube_transcripts.config import Config
from rag_youtube_transcripts.logger import logger
from rag_youtube_transcripts.utils import (
    create_bm25_dataset,
    encode_transcripts,
    fetch_transcripts
)


@logger.catch
def main() -> None:
    """Fetches video transcripts and corresponding metadata from a list of YouTube
    channel IDs, splits the transcripts into contextualized chunks, generates their
    embeddings, and writes the resulting data to `./artifacts/data/transcripts.parquet`
    and `./artifacts/data/embeddings.parquet`.
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
            # update `./artifacts/data/transcripts.parquet`
            (
                pl.concat((data.lazy(), pl.scan_parquet(Config.Paths.transcripts)), how="vertical")
                .sort(by=["creation_date", "video_id"], descending=[True, False])
                .sink_parquet(Config.Paths.transcripts)
            )

            # update `./artifacts/data/embeddings.parquet`
            logger.info("Starting the embedding process...")
            start: float = time.perf_counter()
            (
                pl.concat(
                    (
                        data.pipe(encode_transcripts).lazy(),
                        pl.scan_parquet(Config.Paths.embeddings)
                    ),
                    how="vertical"
                )
                .sort("video_id", "chunk_index")
                .sink_parquet(Config.Paths.embeddings)
            )

            # update `./artifacts/data/bm25.parquet`
            create_bm25_dataset()
            logger.info(
                f"Finished! It took ~{((time.perf_counter() - start)/60):.2f} minutes to generate "
                f"embeddings for {len(data):_} YouTube video transcripts."
            )
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
