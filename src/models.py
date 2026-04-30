"""Pydantic data models for the When Algorithms Meet Artists analysis pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Tier 1 — Input Data Models
# ---------------------------------------------------------------------------

VALID_MEDIA_TYPES = {"article", "audio", "paper", "video"}
VALID_SPECIFIC_TYPES = {"TED", "blog", "interview", "journal", "opinion", "panel", "solo"}
VALID_QUESTION_GROUPS = {"compensation", "ownership", "threat", "transparency", "utility"}
VALID_LIKERT_LEVELS = {
    "strongly_disagree", "disagree", "neutral", "agree", "strongly_agree",
}

GOVERNANCE_THEMES = {"ownership", "transparency", "compensation"}
AFFECTIVE_THEMES = {"threat", "utility"}


class PublicChunk(BaseModel):
    """A single row from the public discourse corpus."""

    line_number: int = Field(ge=0)
    year: int = Field(ge=2013, le=2025)
    article_name: str = Field(min_length=1)
    media_type: str
    specific_type: str
    lexical_diversity: float = Field(ge=0.0, le=1.0)
    section_id: int = Field(ge=0)
    text_og: str = Field(min_length=1)
    text_phrase_norm: str = Field(min_length=1)
    chunk_text_norm: str = Field(min_length=1)
    chunk_word_count_norm: int = Field(ge=1)
    chunk_text_clean: str = Field(min_length=1)
    chunk_word_count_clean: int = Field(ge=1)
    chunk_text_lexical: str = Field(min_length=1)
    chunk_word_count_lexical: int = Field(ge=1)

    @field_validator("media_type", mode="before")
    @classmethod
    def normalize_media_type(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in VALID_MEDIA_TYPES:
            raise ValueError(
                f"Invalid media_type '{v}'. Must be one of {sorted(VALID_MEDIA_TYPES)}"
            )
        return v

    @field_validator("specific_type", mode="before")
    @classmethod
    def normalize_specific_type(cls, v: str) -> str:
        v = str(v).strip()
        if v == "jounal":
            v = "journal"
        return v


class ArtistProbe(BaseModel):
    """A single artist perspective probe from the filtered survey data."""

    respondent_id: int = Field(ge=0)
    Artist: str
    Art_practice: str
    Purchase_art: str = ""
    Professional_artist: str = ""
    AI_models_familiarity: str = ""
    Used_AI_art_models: str = ""
    compensation: str = ""
    Age: str = ""
    POC: str = ""
    Gender_identity: str = ""
    Country: str = ""
    question_group: str
    perspective_text: str = Field(min_length=1)

    @field_validator("question_group", mode="before")
    @classmethod
    def validate_question_group(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in VALID_QUESTION_GROUPS:
            raise ValueError(
                f"Invalid question_group '{v}'. Must be one of {sorted(VALID_QUESTION_GROUPS)}"
            )
        return v

    @field_validator(
        "Artist", "Art_practice", "Purchase_art", "Professional_artist",
        "AI_models_familiarity", "Used_AI_art_models", "compensation",
        "Age", "POC", "Gender_identity", "Country",
        mode="before",
    )
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip()
        return str(v) if v is not None else ""


class LikertPhrase(BaseModel):
    """A single LLM-generated Likert anchor phrase for public probe extraction."""

    id: str = Field(min_length=1)
    theme: str
    likert: str
    style: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("theme", mode="before")
    @classmethod
    def validate_theme(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in VALID_QUESTION_GROUPS:
            raise ValueError(f"Invalid theme '{v}'. Must be one of {sorted(VALID_QUESTION_GROUPS)}")
        return v

    @field_validator("likert", mode="before")
    @classmethod
    def validate_likert(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in VALID_LIKERT_LEVELS:
            raise ValueError(
                f"Invalid likert level '{v}'. Must be one of {sorted(VALID_LIKERT_LEVELS)}"
            )
        return v


class CorpusMetadata(BaseModel):
    """Aggregate metadata about a loaded corpus."""

    name: str
    n_rows: int = Field(ge=0)
    n_documents: int = Field(ge=0)
    year_min: int | None = None
    year_max: int | None = None
    date_loaded: datetime = Field(default_factory=datetime.now)
    source_path: Path
    columns: list[str]


# ---------------------------------------------------------------------------
# Tier 2 — Configuration Models
# ---------------------------------------------------------------------------

MODEL_DIM_MAP = {
    "intfloat/e5-large-v2": 1024,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "all-MiniLM-L6-v2": 384,
}


class EmbeddingConfig(BaseModel):
    """Configuration for sentence embedding generation."""

    model_name: str = "intfloat/e5-large-v2"
    embedding_dim: int = 1024
    batch_size: int = Field(default=32, ge=1)
    normalize_l2: bool = True
    text_column: str = "chunk_text_clean"
    prefix: str | None = Field(
        default="query: ",
        description="Prefix prepended to texts before encoding. "
        "For e5-large-v2, 'query: ' is recommended for clustering tasks.",
    )

    @model_validator(mode="after")
    def check_dim_matches_model(self) -> EmbeddingConfig:
        expected = MODEL_DIM_MAP.get(self.model_name)
        if expected is not None and self.embedding_dim != expected:
            raise ValueError(
                f"Model '{self.model_name}' produces {expected}-dim embeddings, "
                f"but embedding_dim is set to {self.embedding_dim}"
            )
        return self


class UMAPConfig(BaseModel):
    """Configuration for consensus UMAP."""

    n_components: int = Field(default=8, ge=2, le=50)
    n_neighbors: int = Field(default=27, ge=2)
    min_dist: float = Field(default=0.1, ge=0.0, le=1.0)
    metric: str = "cosine"
    seeds: list[int] = Field(
        default_factory=lambda: [
            137, 85, 127, 59, 195, 243, 170, 77, 186, 79,
            69, 42, 240, 105, 199, 91, 151, 82, 177, 234,
            46, 101, 34, 175, 108, 81, 176, 241, 20, 53,
        ]
    )
    consensus_method: Literal["distance_average", "procrustes_average"] = "distance_average"
    distance_metric_lowdim: str = "euclidean"

    @field_validator("seeds", mode="after")
    @classmethod
    def require_multiple_seeds(cls, v: list[int]) -> list[int]:
        if len(v) < 3:
            raise ValueError(f"Consensus requires at least 3 seeds, got {len(v)}")
        return v


class ClusterConfig(BaseModel):
    """Configuration for clustering."""

    method: Literal["hdbscan", "kmeans"] = "kmeans"
    n_clusters: int | None = Field(default=20, description="For kmeans; ignored for HDBSCAN")
    min_cluster_size: int = Field(default=10, ge=2)
    min_samples: int = Field(default=5, ge=1)
    random_state: int = 42


class ProjectionConfig(BaseModel):
    """Configuration for the MLP projection head."""

    hidden_layer_sizes: tuple[int, ...] = (1024, 512, 256, 128, 64)
    activation: str = "relu"
    alpha: float = Field(default=0.0001, ge=0.0)
    learning_rate_init: float = Field(default=0.001, gt=0.0)
    max_iter: int = Field(default=1000, ge=1)
    early_stopping: bool = True
    validation_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_size: float = Field(default=0.15, ge=0.0, lt=1.0)
    random_state: int = 42


class PipelineConfig(BaseModel):
    """Top-level configuration aggregating all sub-configs."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    umap: UMAPConfig = Field(default_factory=UMAPConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    data_dir: Path = Path("data")
    figures_dir: Path = Path("figures")
    random_state: int = 42


# ---------------------------------------------------------------------------
# Tier 3 — Result Models
# ---------------------------------------------------------------------------

class ConsensusResult(BaseModel):
    """Result of a consensus UMAP run."""

    n_samples: int = Field(ge=1)
    n_dimensions: int = Field(ge=2)
    n_seeds: int = Field(ge=1)
    ari_mean_seed_vs_consensus: float = Field(ge=-1.0, le=1.0)
    ari_std_seed_vs_consensus: float = Field(ge=0.0)
    trustworthiness: float = Field(ge=0.0, le=1.0)
    config: UMAPConfig


class ClusterResult(BaseModel):
    """Result of clustering."""

    n_clusters: int = Field(ge=1)
    n_noise_points: int = Field(ge=0)
    noise_fraction: float = Field(ge=0.0, le=1.0)
    silhouette_score: float = Field(ge=-1.0, le=1.0)
    cluster_sizes: dict[int, int]
    method: str
    config: ClusterConfig


class TopicInfo(BaseModel):
    """Information about a single discourse topic."""

    topic_id: int
    label: str
    keywords: list[str]
    macro_theme: str
    n_chunks: int = Field(ge=0)
    percentage: float = Field(ge=0.0, le=100.0)


class SalienceRatio(BaseModel):
    """Salience ratio for a theme at a coverage threshold."""

    theme: str
    threshold: float = Field(description="Cumulative mass threshold (e.g., 0.80, 0.90, 0.95)")
    cluster_set: list[int]
    artist_mass: float = Field(ge=0.0, le=1.0)
    public_mass: float = Field(ge=0.0, le=1.0)
    ratio: float = Field(ge=0.0)
    public_probe_mass: float | None = Field(default=None, ge=0.0, le=1.0)
    public_probe_ratio: float | None = Field(default=None, ge=0.0)

    @field_validator("theme", mode="before")
    @classmethod
    def validate_theme(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in VALID_QUESTION_GROUPS:
            raise ValueError(f"Invalid theme '{v}'")
        return v


class StatisticalTest(BaseModel):
    """Result of a statistical test."""

    test_name: str
    statistic: float
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    effect_size: float | None = None
    effect_size_name: str | None = None
    dof: int | None = Field(default=None, ge=0)
    n_observations: int = Field(ge=1)
    description: str


class ProjectionResult(BaseModel):
    """Result of training the MLP projection head."""

    r2_train: float
    r2_val: float
    knn_preservation: float = Field(ge=0.0, le=1.0)
    n_train: int = Field(ge=1)
    n_val: int = Field(ge=1)
    hidden_layer_sizes: tuple[int, ...]


# ---------------------------------------------------------------------------
# Tier 4 — Top-Level Pipeline Result
# ---------------------------------------------------------------------------

class PipelineResult(BaseModel):
    """Aggregated result of the full analysis pipeline."""

    config: PipelineConfig
    corpus_metadata: dict[str, CorpusMetadata]
    consensus: ConsensusResult
    clustering: ClusterResult
    topics: list[TopicInfo]
    projection: ProjectionResult
    salience_ratios: list[SalienceRatio]
    statistical_tests: list[StatisticalTest]
    figure_paths: list[Path] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
