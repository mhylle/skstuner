"""Quality validation for synthetic clinical notes"""

from typing import List, Dict, Optional
import re
import logging
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)

# Danish language indicators
DANISH_COMMON_WORDS = {
    "og", "er", "med", "til", "af", "på", "i", "for", "ikke", "en", "det", "at",
    "der", "som", "har", "blev", "var", "efter", "ved", "fra", "over", "uden",
    "patient", "behandling", "diagnose", "symptomer", "undersøgelse",
}

# Medical terminology indicators (Danish)
DANISH_MEDICAL_TERMS = {
    "patient", "diagnose", "behandling", "symptom", "undersøgelse", "medicin",
    "smerter", "tilstand", "blod", "sygdom", "operation", "hospital", "læge",
    "klinik", "terapi", "prognose", "akut", "kronisk", "infektion",
}

# Quality thresholds
MIN_LENGTH = 50  # Minimum characters for a valid clinical note
MAX_LENGTH = 2000  # Maximum reasonable length
MIN_WORDS = 10  # Minimum number of words
MIN_DANISH_WORD_RATIO = 0.3  # At least 30% Danish common words
MIN_MEDICAL_TERM_COUNT = 1  # At least 1 medical term


@dataclass
class QualityScore:
    """Quality score for a clinical note"""

    text: str
    length_score: float  # 0-1
    language_score: float  # 0-1
    medical_relevance_score: float  # 0-1
    overall_score: float  # 0-1
    passed: bool
    issues: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "length_score": round(self.length_score, 3),
            "language_score": round(self.language_score, 3),
            "medical_relevance_score": round(self.medical_relevance_score, 3),
            "overall_score": round(self.overall_score, 3),
            "passed": self.passed,
            "issues": self.issues,
        }


class QualityValidator:
    """Validates quality of synthetic clinical notes"""

    def __init__(
        self,
        min_length: int = MIN_LENGTH,
        max_length: int = MAX_LENGTH,
        min_words: int = MIN_WORDS,
        min_danish_ratio: float = MIN_DANISH_WORD_RATIO,
        min_medical_terms: int = MIN_MEDICAL_TERM_COUNT,
        quality_threshold: float = 0.5,
    ):
        """
        Initialize quality validator

        Args:
            min_length: Minimum character length
            max_length: Maximum character length
            min_words: Minimum number of words
            min_danish_ratio: Minimum ratio of Danish common words
            min_medical_terms: Minimum number of medical terms
            quality_threshold: Minimum overall score to pass (0-1)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.min_danish_ratio = min_danish_ratio
        self.min_medical_terms = min_medical_terms
        self.quality_threshold = quality_threshold

    def validate(self, text: str, code_description: Optional[str] = None) -> QualityScore:
        """
        Validate quality of a clinical note

        Args:
            text: Clinical note text to validate
            code_description: Optional SKS code description for relevance checking

        Returns:
            QualityScore object with detailed scores
        """
        issues = []

        # Length validation
        length_score = self._validate_length(text, issues)

        # Language validation (Danish detection)
        language_score = self._validate_language(text, issues)

        # Medical relevance validation
        medical_relevance_score = self._validate_medical_relevance(
            text, code_description, issues
        )

        # Calculate overall score (weighted average)
        overall_score = (
            length_score * 0.2 + language_score * 0.4 + medical_relevance_score * 0.4
        )

        # Determine if passed
        passed = overall_score >= self.quality_threshold and len(issues) == 0

        return QualityScore(
            text=text,
            length_score=length_score,
            language_score=language_score,
            medical_relevance_score=medical_relevance_score,
            overall_score=overall_score,
            passed=passed,
            issues=issues,
        )

    def _validate_length(self, text: str, issues: List[str]) -> float:
        """Validate text length"""
        length = len(text)
        word_count = len(text.split())

        if length < self.min_length:
            issues.append(f"Text too short: {length} chars (min: {self.min_length})")
            return 0.0

        if length > self.max_length:
            issues.append(f"Text too long: {length} chars (max: {self.max_length})")
            return 0.5

        if word_count < self.min_words:
            issues.append(f"Too few words: {word_count} (min: {self.min_words})")
            return 0.3

        # Score based on optimal length range (100-500 chars)
        if 100 <= length <= 500:
            return 1.0
        elif length < 100:
            return length / 100
        else:
            # Gradually decrease score for very long texts
            return max(0.7, 1.0 - (length - 500) / 2000)

    def _validate_language(self, text: str, issues: List[str]) -> float:
        """Validate Danish language indicators"""
        # Normalize text
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        if not words:
            issues.append("No words found in text")
            return 0.0

        # Check for Danish common words
        danish_word_count = sum(1 for word in words if word in DANISH_COMMON_WORDS)
        danish_ratio = danish_word_count / len(words)

        if danish_ratio < self.min_danish_ratio:
            issues.append(
                f"Low Danish word ratio: {danish_ratio:.2f} (min: {self.min_danish_ratio})"
            )
            return danish_ratio / self.min_danish_ratio

        # Check for non-Danish characters (too many English/other language indicators)
        # Simple heuristic: check for Danish-specific characters
        has_danish_chars = any(c in text for c in ["æ", "ø", "å", "Æ", "Ø", "Å"])

        if has_danish_chars:
            return 1.0
        else:
            # Still might be Danish, just without special chars
            return min(0.9, danish_ratio / self.min_danish_ratio)

    def _validate_medical_relevance(
        self, text: str, code_description: Optional[str], issues: List[str]
    ) -> float:
        """Validate medical relevance"""
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        # Check for medical terminology
        medical_term_count = sum(1 for word in words if word in DANISH_MEDICAL_TERMS)

        if medical_term_count < self.min_medical_terms:
            issues.append(
                f"Low medical term count: {medical_term_count} (min: {self.min_medical_terms})"
            )
            return 0.3

        # Basic relevance score based on medical term density
        medical_term_ratio = medical_term_count / len(words) if words else 0

        # If code description provided, check for keyword overlap
        relevance_score = min(1.0, medical_term_ratio * 10)  # Scale up ratio

        if code_description:
            desc_words = set(re.findall(r"\b\w+\b", code_description.lower()))
            overlap = len(words & desc_words)
            if overlap > 0:
                relevance_score = min(1.0, relevance_score + overlap * 0.1)

        return relevance_score

    def validate_dataset(
        self, examples: List[Dict], show_progress: bool = True
    ) -> Dict[str, any]:
        """
        Validate entire dataset

        Args:
            examples: List of example dicts with 'text' and optional 'description'
            show_progress: Whether to log progress

        Returns:
            Dictionary with validation statistics
        """
        passed = 0
        failed = 0
        scores = []
        failed_examples = []

        for i, example in enumerate(examples):
            text = example.get("text", "")
            description = example.get("description")

            quality = self.validate(text, description)
            scores.append(quality.overall_score)

            if quality.passed:
                passed += 1
            else:
                failed += 1
                failed_examples.append(
                    {
                        "index": i,
                        "label": example.get("label"),
                        "score": quality.overall_score,
                        "issues": quality.issues,
                        "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    }
                )

            if show_progress and (i + 1) % 100 == 0:
                logger.info(f"Validated {i + 1}/{len(examples)} examples")

        avg_score = sum(scores) / len(scores) if scores else 0

        stats = {
            "total_examples": len(examples),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(examples) if examples else 0,
            "average_score": avg_score,
            "failed_examples": failed_examples[:20],  # Limit to first 20
        }

        logger.info(
            f"Validation complete: {passed}/{len(examples)} passed "
            f"({stats['pass_rate']:.1%}), avg score: {avg_score:.3f}"
        )

        return stats

    def filter_dataset(
        self, examples: List[Dict], show_progress: bool = True
    ) -> List[Dict]:
        """
        Filter dataset to keep only high-quality examples

        Args:
            examples: List of example dicts
            show_progress: Whether to log progress

        Returns:
            Filtered list of examples
        """
        filtered = []

        for i, example in enumerate(examples):
            text = example.get("text", "")
            description = example.get("description")

            quality = self.validate(text, description)

            if quality.passed:
                # Add quality score to example
                example["quality_score"] = quality.overall_score
                filtered.append(example)

            if show_progress and (i + 1) % 100 == 0:
                logger.info(
                    f"Filtered {i + 1}/{len(examples)} examples "
                    f"({len(filtered)} kept, {i + 1 - len(filtered)} removed)"
                )

        logger.info(
            f"Filtering complete: kept {len(filtered)}/{len(examples)} examples "
            f"({len(filtered)/len(examples):.1%})"
        )

        return filtered


def calculate_diversity_metrics(examples: List[Dict]) -> Dict:
    """
    Calculate diversity metrics for a dataset

    Args:
        examples: List of example dicts with 'text' field

    Returns:
        Dictionary with diversity metrics
    """
    if not examples:
        return {"error": "Empty dataset"}

    texts = [ex.get("text", "") for ex in examples]

    # Calculate unique text ratio
    unique_texts = len(set(texts))
    unique_ratio = unique_texts / len(texts)

    # Calculate average text length variance
    lengths = [len(text) for text in texts]
    avg_length = sum(lengths) / len(lengths)
    length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)

    # Calculate vocabulary diversity
    all_words = []
    for text in texts:
        words = re.findall(r"\b\w+\b", text.lower())
        all_words.extend(words)

    vocab_size = len(set(all_words))
    total_words = len(all_words)
    vocab_ratio = vocab_size / total_words if total_words > 0 else 0

    # Calculate most common words (excluding very common Danish words)
    word_freq = Counter(all_words)
    # Remove very common words
    for common_word in DANISH_COMMON_WORDS:
        word_freq.pop(common_word, None)

    most_common = word_freq.most_common(20)

    return {
        "total_examples": len(examples),
        "unique_texts": unique_texts,
        "unique_ratio": unique_ratio,
        "avg_text_length": avg_length,
        "length_variance": length_variance,
        "vocabulary_size": vocab_size,
        "total_words": total_words,
        "vocabulary_diversity_ratio": vocab_ratio,
        "most_common_words": most_common,
    }
