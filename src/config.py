"""This module sets up the project's configuration."""

from pathlib import Path, PosixPath

from omegaconf import DictConfig, OmegaConf


class Config:
    """A class to encapsulate the project's configuration.

    Attributes:
        project_dir (PosixPath): Project's root directory.
        artifacts_dir (PosixPath): Artifacts directory, ./artifacts/.
        data_dir (PosixPath): Data directory, ./artifacts/data/.
        models_dir (PosixPath): Models directory, ./artifacts/models/.
        logs_dir (PosixPath): Logs directory, ./logs/.
        env (PosixPath): .env file, ./.env.
        params (PosixPath): Parameters file, ./params.yaml.
        transcripts (PosixPath): YouTube video transcripts' data,
        ./artifacts/data/transcripts.parquet.
        embeddings: (PosixPath): YouTube video transcripts' embeddings data,
        ./artifacts/data/embeddings.parquet.
        youtube_data_api: str = YouTube data API.
    """
    project_dir: PosixPath = Path(__file__).parent.parent.absolute()
    artifacts_dir: PosixPath = project_dir / "artifacts"
    data_dir: PosixPath = artifacts_dir / "data"
    models_dir: PosixPath = artifacts_dir / "models"
    logs_dir: PosixPath = project_dir / "logs"
    env: PosixPath = project_dir / ".env"
    params: PosixPath = project_dir / "params.yaml"
    transcripts: PosixPath = data_dir / "transcripts.parquet"
    embeddings: PosixPath = data_dir / "embeddings.parquet"
    youtube_data_api: str = "https://www.googleapis.com/youtube/v3/search"

    @classmethod
    def load_params(cls, key: str) -> DictConfig:
        """Loads cls.params as a DictConfig object.

        Args:
            key (str): Unique identifier that's used to retrieve specific user-defined
            key-value pairs.

        Returns:
            DictConfig: Dictionary-like object with user-defined key-value pairs.
        """
        try:
            return OmegaConf.load(cls.params).get(key)
        except Exception as e:
            raise e
