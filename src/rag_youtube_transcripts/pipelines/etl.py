"""This script executes the YouTube video transcripts ETL pipeline."""

import time
from pathlib import Path

import polars as pl
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rag_youtube_transcripts.config import Config
from rag_youtube_transcripts.logger import logger
from rag_youtube_transcripts.utils import create_bm25_dataset, encode_transcripts, fetch_transcripts


PARAMS: DictConfig = Config.load_params(Path(__file__).stem)


def sink_atomically(plan: pl.LazyFrame, path: Path) -> None:
    """Streams plan to a sibling temp file, then atomically swaps it into place."""
    tmp_path: Path = path.with_name(f"{path.stem}_tmp{path.suffix}")
    try:
        plan.sink_parquet(tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


@logger.catch
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
        dfs: list[pl.DataFrame] = Parallel(n_jobs=-1)(
            delayed(fetch_transcripts)(youtube_channel_id)
            for youtube_channel_id in tqdm(
                iterable=youtube_channel_ids,
                desc="Fetching YouTube video transcripts",
                total=len(youtube_channel_ids),
                unit="YouTube Channel ID"
            )
        )
        data: pl.DataFrame = pl.concat(dfs, how="vertical")
        if data.is_empty():
            logger.info("There are no new transcripts. Skipping the embedding process.")
        else:
            logger.info("Starting the indexing process...")
            start: float = time.perf_counter()

            # update `./artifacts/data/embeddings.parquet`
            path: Path = Config.Paths.embeddings
            plan: pl.LazyFrame = (
                pl.concat(
                    (data.pipe(encode_transcripts).lazy(), pl.scan_parquet(path)),
                    how="vertical"
                )
                .sort("video_id", "chunk_index")
            )
            sink_atomically(plan, path)

            # update `./artifacts/data/bm25.parquet`
            create_bm25_dataset()

            # update `./artifacts/data/transcripts.parquet`
            path = Config.Paths.transcripts
            plan = (
                pl.concat((data.lazy(), pl.scan_parquet(path)), how="vertical")
                .sort("creation_date", "video_id", descending=[True, False])
            )
            sink_atomically(plan, path)

            logger.info(
                f"Finished! It took ~{((time.perf_counter() - start)/60):.2f} minutes to index "
                f"{data.height:_} YouTube video transcripts."
            )
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
