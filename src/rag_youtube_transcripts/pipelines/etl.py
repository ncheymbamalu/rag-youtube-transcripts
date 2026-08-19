"""This script executes the YouTube video transcripts ETL pipeline."""

import time
from pathlib import Path

import polars as pl
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rag_youtube_transcripts.config import Config
from rag_youtube_transcripts.index import create_bm25_dataset, encode_transcripts
from rag_youtube_transcripts.ingest import fetch_transcripts
from rag_youtube_transcripts.logger import logger


PARAMS: DictConfig = Config.load_params(Path(__file__).stem)


def sink_atomically(plan: pl.LazyFrame, path: Path) -> None:
    """Streams plan to a sibling temp file, then atomically swaps it into place."""
    tmp_path: Path = path.with_name(f"{path.stem}_tmp{path.suffix}")
    try:
        plan.sink_parquet(tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


@logger.catch(reraise=True)
def main() -> None:
    """Fetches video transcripts and corresponding metadata from a list of YouTube
    channel IDs, splits the transcripts into contextualized chunks, generates their
    embeddings, and updates the following files, `./artifacts/data/embeddings.parquet`
    `./artifacts/data/bm25.parquet`, and `./artifacts/data/transcripts.parquet`,
    in that order.
    """
    try:
        # fetch the YouTube video transcripts
        youtube_channel_ids: list[str] = OmegaConf.to_container(PARAMS.youtube_channel_ids)
        n_channel_ids: int = len(youtube_channel_ids)
        results: list[pl.DataFrame | None] = Parallel(n_jobs=-1)(
            delayed(fetch_transcripts)(youtube_channel_id)
            for youtube_channel_id in tqdm(
                iterable=youtube_channel_ids,
                desc="Fetching YouTube video transcripts",
                total=n_channel_ids,
                unit="YouTube Channel ID"
            )
        )
        failed: int = sum(1 for result in results if result is None)
        dfs: list[pl.DataFrame] = [
            result for result in results if isinstance(result, pl.DataFrame)
        ]
        if not dfs:
            raise RuntimeError(
                f"All {n_channel_ids} YouTube channels failed to fetch. "
                "Check YOUTUBE_DATA_API_KEY and network connectivity."
            )
        if failed:
            logger.warning(f"<red>{failed}</> of {n_channel_ids} channels failed to fetch.")
        data: pl.DataFrame = pl.concat(dfs, how="vertical")
        if data.is_empty():
            logger.info("There are no new transcripts. Skipping the embedding process.")
        else:
            logger.info("Starting the indexing process...")
            start: float = time.perf_counter()

            # update `./artifacts/data/embeddings.parquet`
            (
                pl.concat(
                    [
                        data.pipe(encode_transcripts).lazy(),
                        pl.scan_parquet(Config.Paths.embeddings)
                    ],
                    how="vertical"
                )
                .sort("video_id", "chunk_index")
                .pipe(sink_atomically, Config.Paths.embeddings)
            )

            # update `./artifacts/data/bm25.parquet`
            create_bm25_dataset().pipe(sink_atomically, Config.Paths.bm25_data)

            # update `./artifacts/data/transcripts.parquet`
            (
                pl.concat(
                    [data.lazy(), pl.scan_parquet(Config.Paths.transcripts)],
                    how="vertical"
                )
                .sort("creation_date", "video_id", descending=[True, False])
                .pipe(sink_atomically, Config.Paths.transcripts)
            )

            logger.info(
                f"Finished! It took ~{((time.perf_counter() - start)/60):.2f} minutes to index "
                f"{data.height:_} YouTube video transcripts."
            )
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
