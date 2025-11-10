"""Tests for quality validator"""

import pytest
from skstuner.data.quality_validator import (
    QualityValidator,
    QualityScore,
    calculate_diversity_metrics,
)


@pytest.fixture
def validator():
    """Fixture for quality validator"""
    return QualityValidator(quality_threshold=0.5)


def test_quality_validator_init():
    """Test validator initialization"""
    validator = QualityValidator(
        min_length=100, max_length=1000, quality_threshold=0.7
    )
    assert validator.min_length == 100
    assert validator.max_length == 1000
    assert validator.quality_threshold == 0.7


def test_validate_good_danish_text(validator):
    """Test validation of good Danish clinical text"""
    text = (
        "Patient har smerter i højre ben efter fald. "
        "Undersøgelse viser hævelse og ømhed. "
        "Diagnose: Forstrækning af muskler. "
        "Behandling: Hvile og smertestillende medicin."
    )

    quality = validator.validate(text)

    assert isinstance(quality, QualityScore)
    assert quality.passed is True
    assert quality.overall_score > 0.5
    assert quality.language_score > 0
    assert quality.medical_relevance_score > 0


def test_validate_short_text(validator):
    """Test validation of too short text"""
    text = "Patient."

    quality = validator.validate(text)

    assert quality.passed is False
    assert any("short" in issue.lower() for issue in quality.issues)
    assert quality.length_score < 1.0


def test_validate_long_text(validator):
    """Test validation of very long text"""
    text = "Patient " * 500  # Very long repetitive text

    quality = validator.validate(text)

    # Should still pass but with reduced length score
    assert quality.length_score <= 1.0


def test_validate_non_danish_text(validator):
    """Test validation of non-Danish text"""
    text = (
        "The patient presents with severe headache and nausea. "
        "Physical examination shows normal vital signs. "
        "Treatment plan includes pain medication and observation."
    )

    quality = validator.validate(text)

    # Should have low language score
    assert quality.language_score < 0.5


def test_validate_no_medical_terms(validator):
    """Test validation of text without medical terms"""
    text = (
        "Dette er en lang tekst på dansk som ikke handler om medicin eller sundhed. "
        "Den er bare en almindelig tekst med mange ord og sætninger."
    )

    quality = validator.validate(text)

    # Should have low medical relevance score
    assert quality.medical_relevance_score < 1.0


def test_validate_with_code_description(validator):
    """Test validation with code description for relevance"""
    text = "Patient har diabetes og høj blodsukker"
    description = "diabetes type 2"

    quality = validator.validate(text, description)

    # Should detect relevance based on keyword overlap
    assert quality.medical_relevance_score > 0


def test_quality_score_to_dict():
    """Test QualityScore to_dict method"""
    score = QualityScore(
        text="test",
        length_score=0.8,
        language_score=0.9,
        medical_relevance_score=0.7,
        overall_score=0.8,
        passed=True,
        issues=[],
    )

    result = score.to_dict()

    assert result["length_score"] == 0.8
    assert result["language_score"] == 0.9
    assert result["medical_relevance_score"] == 0.7
    assert result["overall_score"] == 0.8
    assert result["passed"] is True
    assert result["issues"] == []


def test_validate_dataset(validator):
    """Test dataset validation"""
    dataset = [
        {
            "text": "Patient har smerter i ben og arm efter fald. Undersøgelse viser "
                   "blå mærker og ømhed. Diagnose er forstrækning og behandling er hvile "
                   "samt smertestillende medicin ved behov.",
            "label": "D50",
            "description": "smerter",
        },
        {
            "text": "Kort",  # Too short
            "label": "D51",
            "description": "test",
        },
        {
            "text": "Patient undersøges grundigt for tegn på infektion efter operation. "
                   "Blodprøver tages og patienten får antibiotika behandling som forebyggelse. "
                   "Tilstanden er stabil.",
            "label": "D52",
            "description": "infektion",
        },
    ]

    stats = validator.validate_dataset(dataset, show_progress=False)

    assert stats["total_examples"] == 3
    assert stats["passed"] >= 1  # At least some should pass
    assert stats["failed"] >= 1  # At least one should fail (the short one)
    assert 0 <= stats["pass_rate"] <= 1
    assert 0 <= stats["average_score"] <= 1
    assert "failed_examples" in stats


def test_filter_dataset(validator):
    """Test dataset filtering"""
    dataset = [
        {
            "text": "Patient har smerter i ben og får behandling med smertestillende medicin.",
            "label": "D50",
        },
        {
            "text": "x",  # Too short, should be filtered
            "label": "D51",
        },
        {
            "text": "Patient undersøges grundigt og diagnose viser infektion som behandles.",
            "label": "D52",
        },
    ]

    filtered = validator.filter_dataset(dataset, show_progress=False)

    # Should filter out the too-short example
    assert len(filtered) < len(dataset)
    assert all("quality_score" in ex for ex in filtered)


def test_calculate_diversity_metrics():
    """Test diversity metrics calculation"""
    dataset = [
        {"text": "Patient har smerter og får behandling."},
        {"text": "Patient har smerter og får behandling."},  # Duplicate
        {"text": "Undersøgelse viser normal tilstand."},
    ]

    diversity = calculate_diversity_metrics(dataset)

    assert diversity["total_examples"] == 3
    assert diversity["unique_texts"] == 2  # One duplicate
    assert diversity["unique_ratio"] < 1.0
    assert diversity["avg_text_length"] > 0
    assert diversity["vocabulary_size"] > 0
    assert diversity["total_words"] > 0
    assert 0 <= diversity["vocabulary_diversity_ratio"] <= 1
    assert "most_common_words" in diversity


def test_calculate_diversity_empty_dataset():
    """Test diversity metrics with empty dataset"""
    diversity = calculate_diversity_metrics([])

    assert "error" in diversity


def test_validate_with_danish_characters(validator):
    """Test validation recognizes Danish special characters"""
    text = (
        "Patienten har været syg i flere dage med høj feber. "
        "Undersøgelsen viser tegn på infektion. "
        "Behandling: Antibiotika og hvile."
    )

    quality = validator.validate(text)

    # Should give high language score due to Danish chars (æ, ø, å)
    assert quality.language_score > 0.8


def test_validator_custom_thresholds():
    """Test validator with custom thresholds"""
    strict_validator = QualityValidator(
        min_length=100,
        min_words=20,
        min_danish_ratio=0.5,
        quality_threshold=0.8,
    )

    text = "Patient har smerter."

    quality = strict_validator.validate(text)

    # With stricter thresholds, should be more likely to fail
    assert quality.overall_score < 1.0


def test_validate_medical_terminology():
    """Test detection of Danish medical terms"""
    text = (
        "Patient kommer med akutte smerter. "
        "Diagnose efter undersøgelse: Infektion. "
        "Behandling: Antibiotika og smertestillende medicin ved behov."
    )

    quality = QualityValidator().validate(text)

    # Should detect multiple medical terms
    assert quality.medical_relevance_score > 0.5
