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
            dirs: list[Path] = [
                cls.Paths.artifacts_dir,
                cls.Paths.data_dir,
                cls.Paths.models_dir,
            ]
            for _dir in dirs:
                _dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise e

    @classmethod
    def load_params(cls, module: str) -> DictConfig:
        """Loads module-specific parameters from `./params.yaml`.

        Args:
            module (str): Name of a user-defined module.

        Returns:
            DictConfig: Module-specific parameters in the form of user-defined
            key-value pairs.
        """
        try:
            params: DictConfig = OmegaConf.load(cls.Paths.params)
            return params.get(module)
        except Exception as e:
            raise e
