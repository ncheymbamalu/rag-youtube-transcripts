"""This module sets up the project's configuration."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


class Config:
    """Central configuration namespace for the project."""

    class Paths:
        """Defines and manages the absolute paths for the project's main files and directories.

        Attributes:
            home_dir (Path): Project's home directory.
            artifacts_dir (Path): Artifacts directory, ./artifacts/.
            data_dir (Path): Data directory, ./artifacts/data/.
            models_dir (Path): Models directory, ./artifacts/models/.
            logs_dir (Path): Logs directory, ./logs/.
            env (Path): .env file, ./.env.
            params (Path): Parameters file, ./params.yaml.
            transcripts (Path): YouTube video transcripts' data,
            ./artifacts/data/transcripts.parquet.
            embeddings: (Path): YouTube video transcripts' embeddings data,
            ./artifacts/data/embeddings.parquet.
            bm25_data (Path): BM25 metadata, ./artifacts/data/bm25_data.parquet.
        """
        home_dir: Path = Path(__file__).parent.parent.parent.resolve()
        artifacts_dir: Path = home_dir / "artifacts"
        data_dir: Path = artifacts_dir / "data"
        models_dir: Path = artifacts_dir / "models"
        logs_dir: Path = home_dir / "logs"
        env: Path = home_dir / ".env"
        params: Path = home_dir / "params.yaml"
        transcripts: Path = data_dir / "transcripts.parquet"
        embeddings: Path = data_dir / "embeddings.parquet"
        bm25_data: Path = data_dir / "bm25.parquet"

    @classmethod
    def make_dirs(cls) -> None:
        """Creates the project's main directories if they do not exist."""
        try:
            directories: list[Path] = [
                cls.Paths.artifacts_dir,
                cls.Paths.data_dir,
                cls.Paths.models_dir,
            ]
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise e

    @classmethod
    def load_params(cls, key: str) -> DictConfig:
        """Loads cls.Paths.params as a DictConfig object.

        Args:
            key (str): Unique identifier that's used to retrieve specific user-defined
            key-value pairs.

        Returns:
            DictConfig: Dictionary-like object with user-defined key-value pairs.
        """
        try:
            return OmegaConf.load(cls.Paths.params).get(key)
        except Exception as e:
            raise e
