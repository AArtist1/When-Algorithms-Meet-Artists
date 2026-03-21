"""Tests for Pydantic data model validation."""

import pytest
from pydantic import ValidationError

from src.models import (
    ArtistProbe,
    ClusterConfig,
    EmbeddingConfig,
    LikertPhrase,
    PipelineConfig,
    PublicChunk,
    SalienceRatio,
    StatisticalTest,
    UMAPConfig,
)

VALID_PUBLIC_CHUNK = {
    "line_number": 1,
    "year": 2023,
    "article_name": "AI Art and Ethics",
    "media_type": "article",
    "specific_type": "opinion",
    "lexical_diversity": 0.75,
    "section_id": 1,
    "text_og": "Original text here",
    "text_phrase_norm": "normalized text here",
    "chunk_text_norm": "norm chunk",
    "chunk_word_count_norm": 10,
    "chunk_text_clean": "clean chunk",
    "chunk_word_count_clean": 8,
    "chunk_text_lexical": "lexical chunk",
    "chunk_word_count_lexical": 9,
}

VALID_ARTIST_PROBE = {
    "respondent_id": 42,
    "Artist": "Yes",
    "Art_practice": "painting",
    "question_group": "threat",
    "perspective_text": "I agree that AI art models are a threat to art workers",
}

VALID_LIKERT = {
    "id": "utility_strongly_agree_01",
    "theme": "utility",
    "likert": "strongly_agree",
    "style": "blog_opinion",
    "text": "AI tools greatly enhance creative workflows",
}


@pytest.mark.models
class TestPublicChunk:
    def test_valid_construction(self):
        chunk = PublicChunk(**VALID_PUBLIC_CHUNK)
        assert chunk.year == 2023, f"FAILED: Expected year 2023, got {chunk.year}"
        assert chunk.media_type == "article"

    def test_rejects_year_before_2013(self):
        data = {**VALID_PUBLIC_CHUNK, "year": 2000}
        with pytest.raises(ValidationError):
            PublicChunk(**data)

    def test_rejects_year_after_2025(self):
        data = {**VALID_PUBLIC_CHUNK, "year": 2030}
        with pytest.raises(ValidationError):
            PublicChunk(**data)

    def test_rejects_invalid_media_type(self):
        data = {**VALID_PUBLIC_CHUNK, "media_type": "tweet"}
        with pytest.raises(ValidationError):
            PublicChunk(**data)

    def test_normalizes_media_type_whitespace(self):
        data = {**VALID_PUBLIC_CHUNK, "media_type": "video "}
        chunk = PublicChunk(**data)
        assert chunk.media_type == "video", (
            f"FAILED: Expected 'video' after whitespace strip, got '{chunk.media_type}'"
        )

    def test_normalizes_specific_type_typo(self):
        data = {**VALID_PUBLIC_CHUNK, "specific_type": "jounal"}
        chunk = PublicChunk(**data)
        assert chunk.specific_type == "journal", (
            f"FAILED: Expected 'journal' after typo fix, got '{chunk.specific_type}'"
        )

    def test_rejects_empty_text(self):
        data = {**VALID_PUBLIC_CHUNK, "chunk_text_clean": ""}
        with pytest.raises(ValidationError):
            PublicChunk(**data)

    def test_rejects_negative_word_count(self):
        data = {**VALID_PUBLIC_CHUNK, "chunk_word_count_clean": 0}
        with pytest.raises(ValidationError):
            PublicChunk(**data)

    def test_rejects_lexical_diversity_above_1(self):
        data = {**VALID_PUBLIC_CHUNK, "lexical_diversity": 1.5}
        with pytest.raises(ValidationError):
            PublicChunk(**data)


@pytest.mark.models
class TestArtistProbe:
    def test_valid_construction(self):
        probe = ArtistProbe(**VALID_ARTIST_PROBE)
        assert probe.question_group == "threat"

    def test_rejects_invalid_question_group(self):
        data = {**VALID_ARTIST_PROBE, "question_group": "familiarity"}
        with pytest.raises(ValidationError):
            ArtistProbe(**data)

    def test_strips_whitespace(self):
        data = {**VALID_ARTIST_PROBE, "Artist": "Yes "}
        probe = ArtistProbe(**data)
        assert probe.Artist == "Yes", (
            f"FAILED: Expected 'Yes' after strip, got '{probe.Artist}'"
        )

    def test_rejects_empty_perspective_text(self):
        data = {**VALID_ARTIST_PROBE, "perspective_text": ""}
        with pytest.raises(ValidationError):
            ArtistProbe(**data)


@pytest.mark.models
class TestLikertPhrase:
    def test_valid_construction(self):
        phrase = LikertPhrase(**VALID_LIKERT)
        assert phrase.theme == "utility"

    def test_rejects_invalid_theme(self):
        data = {**VALID_LIKERT, "theme": "politics"}
        with pytest.raises(ValidationError):
            LikertPhrase(**data)

    def test_rejects_invalid_likert_level(self):
        data = {**VALID_LIKERT, "likert": "very_agree"}
        with pytest.raises(ValidationError):
            LikertPhrase(**data)


@pytest.mark.models
class TestConfigs:
    def test_embedding_config_validates_dim(self):
        with pytest.raises(ValidationError):
            EmbeddingConfig(model_name="intfloat/e5-large-v2", embedding_dim=768)

    def test_umap_config_rejects_too_few_seeds(self):
        with pytest.raises(ValidationError):
            UMAPConfig(seeds=[42, 7])

    def test_pipeline_config_defaults(self):
        config = PipelineConfig()
        assert config.embedding.model_name == "intfloat/e5-large-v2"
        assert len(config.umap.seeds) == 31, (
            f"FAILED: Expected 31 default seeds, got {len(config.umap.seeds)}"
        )

    def test_salience_ratio_rejects_negative_mass(self):
        with pytest.raises(ValidationError):
            SalienceRatio(
                theme="threat", threshold=0.9, cluster_set=[1],
                artist_mass=-0.1, public_mass=0.5, ratio=1.0,
            )

    def test_statistical_test_rejects_invalid_pvalue(self):
        with pytest.raises(ValidationError):
            StatisticalTest(
                test_name="chi2", statistic=100.0, p_value=1.5,
                n_observations=100, description="test",
            )
