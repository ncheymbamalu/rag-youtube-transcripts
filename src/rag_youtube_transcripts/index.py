"""This module contains functionality for indexing YouTube video transcripts, that is,
splitting each transcript into contextualized chunks, generating each chunk's embedding,
and computing each token's BM25 score with respect to each chunk.
"""

import string
from pathlib import Path

import numpy as np
import polars as pl
from chonkie import TokenChunker
from groq.types.chat import ChatCompletion
from groq.types.chat.chat_completion import Choice as ChatCompletionChoice
from nltk.corpus import stopwords
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from rag_youtube_transcripts.config import Config
from rag_youtube_transcripts.logger import logger
from rag_youtube_transcripts.models import EMBEDDING_MODEL, GROQ_CLIENT


PARAMS: DictConfig = Config.load_params(Path(__file__).stem)


def add_context_to_chunk(
    transcript: str,
    chunk: str,
    llm: str = PARAMS.chunking.llm,
    temperature: float = PARAMS.chunking.temperature,
    max_output_tokens: int = PARAMS.chunking.max_output_tokens,
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
        system_prompt: str = PARAMS.chunking.system_prompt
        user_prompt: str = PARAMS.chunking.user_prompt
        user_prompt = user_prompt.format(transcript=transcript, chunk=chunk)
        reasoning_effort: str = PARAMS.chunking.reasoning_effort
        completion: ChatCompletion = GROQ_CLIENT.chat.completions.create(
            model=llm,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_completion_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort
        )
        choice: ChatCompletionChoice = completion.choices[0]
        if choice.finish_reason == "length" or not choice.message.content:
            logger.warning(
                f"Empty/truncated context (finish_reason={choice.finish_reason}). "
                f"Using original chunk: '{chunk[:80]}...'"
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
        chunk_size: int = PARAMS.encoding.chunk_size
        max_transcript_tokens: int = PARAMS.encoding.max_transcript_tokens
        chunker: TokenChunker = TokenChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=(chunk_size // 10)
        )
        dfs: list[pl.DataFrame] = []
        for video_id, transcript in tqdm(
            iterable=zip(data.get_column("video_id"), data.get_column("transcript"), strict=True),
            desc="Splitting transcripts into contextual chunks and generating their embeddings",
            total=data.height,
            unit="transcript",
            
        ):
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


def create_bm25_dataset(k1: float = 1.5, b: float = 0.75) -> pl.LazyFrame:
    """Creates a dataset that stores each token's BM25 score with repect to
    each contextualized chunk and returns the result as a Polars LazyFrame.
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

        # create the final plan, which contains the BM25 score for each unique ...
        # (video_id, chunk_index, token) group
        plan = (
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
        )
        return plan
    except Exception as e:
        raise e
