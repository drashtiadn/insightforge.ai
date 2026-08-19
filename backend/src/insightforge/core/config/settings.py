"""Application settings from environment variables and optional `.env`."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[4] / ".env"


class Environment(StrEnum):
    """App environment.

    - development: local work on a feature branch
    - production: code on main (merged / deployed)
    """

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Typed app config. Env vars override `.env`, which overrides defaults."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    app_name: str = "insightforge"
    app_version: str = "0.1.0"
    app_env: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_v1_prefix: str = "/api/v1"

    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    cors_expose_headers: list[str] = ["X-Request-ID", "X-Process-Time"]

    secret_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    # LLM (Gemini) — planner / reasoner / report. Heuristics run when unavailable.
    llm_provider: str = "gemini"  # gemini | none
    llm_gemini_model: str = "gemini-2.5-flash"
    llm_timeout_seconds: float = 60.0
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.2

    # LangSmith observability (https://smith.langchain.com)
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "insightforge"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Search providers (Phase 3.2)
    tavily_api_key: SecretStr | None = None
    searxng_base_url: str | None = None
    github_token: SecretStr | None = None
    youtube_api_key: SecretStr | None = None
    reddit_user_agent: str = "insightforge/0.1.0 (research platform)"
    search_default_limit: int = 5
    search_timeout_seconds: float = 15.0
    search_max_workers: int = 4
    search_rate_limit_per_second: float = 2.0
    search_dedupe_enabled: bool = True
    search_scoring_enabled: bool = True
    search_max_documents: int = 20

    # Document processing (Phase 4)
    tesseract_cmd: str | None = None  # optional path to the tesseract binary
    document_cleaning_enabled: bool = True
    document_chunking_enabled: bool = True
    document_chunk_strategy: str = "auto"  # auto | recursive | markdown | semantic
    document_chunk_size: int = 1200
    document_chunk_overlap: int = 150
    document_semantic_threshold: float = 0.25
    document_citation_enabled: bool = True

    # Embeddings (Phase 5.1)
    voyage_api_key: SecretStr | None = None
    embedding_provider: str = "voyage"  # voyage | local
    embedding_voyage_model: str = "voyage-3.5"
    embedding_voyage_dimensions: int | None = None  # optional output_dimension
    embedding_local_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_timeout_seconds: float = 30.0
    embedding_batch_size: int = 32

    # Vector stores (Phase 5.2)
    vector_store: str = "memory"  # memory | qdrant
    vector_dimensions: int = 1024
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "insightforge"
    qdrant_timeout_seconds: float = 30.0
    vector_session_ttl_seconds: float = 3600.0

    # Retrieval (Phase 5.3)
    retrieval_mode: str = "hybrid"  # semantic | bm25 | hybrid
    retrieval_default_limit: int = 8
    retrieval_candidate_multiplier: int = 4
    retrieval_rrf_k: int = 60
    retrieval_bm25_k1: float = 1.5
    retrieval_bm25_b: float = 0.75

    # Reranking (Phase 5.4)
    jina_api_key: SecretStr | None = None
    reranker_provider: str = "cross-encoder"  # cross-encoder | bge | jina
    reranker_cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_bge_model: str = "BAAI/bge-reranker-base"
    reranker_jina_model: str = "jina-reranker-v2-base-multilingual"
    reranker_timeout_seconds: float = 30.0
    reranker_top_n: int | None = None

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @model_validator(mode="after")
    def check_production(self) -> Self:
        if self.is_production:
            if self.debug:
                raise ValueError("DEBUG must be false in production")
            if self.secret_key is None:
                raise ValueError("SECRET_KEY is required in production")
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton for the process."""

    return Settings()
