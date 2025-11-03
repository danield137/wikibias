from dataclasses import dataclass
from typing import Any


@dataclass
class BiasFinding:
    """Standard output schema for text-scanning tools."""

    kind: str  # The type of bias/signal (e.g., 'loaded_language', 'passive_voice_omitted_actor')
    strength: float  # Estimated bias strength (0.0 = weak signal, 1.0 = strong bias)
    text: str  # The exact text span where the bias was found
    offset: tuple[int | None, int | None]  # The [start, end] character index in the original text
    explanation: str  # A clear, concise explanation of the bias


@dataclass
class SourceAnalysis:
    """Standard output schema for source analysis tools."""

    source_id: str  # The source description (e.g., 'Reuters, Oct 31')
    analysis_type: str  # The type of analysis performed (e.g., 'integrity', 'clustering', 'diversity', 'political_bias')
    report: dict[str, Any]  # A custom object containing the specific findings


@dataclass
class IntegrityReport:
    """Report schema for source integrity analysis."""

    source_reliability: float  # Source reliability score (0-1)
    source_bias_score: float  # Source bias score (-1 to 1, where -1 is left-leaning, 1 is right-leaning)
    verification_strength: str  # How well the source verifies the claim: "Full", "Partial", or "None"
    explanation: str  # Detailed explanation of the analysis


@dataclass
class ClusteringReport:
    """Report schema for citation clustering analysis."""

    is_clustered: bool  # Whether the sources cluster around a single original source
    independent_sources: int  # Number of truly independent sources
    total_citations: int  # Total number of citations analyzed
    original_source: str  # The original/primary source if clustering detected
    explanation: str  # Detailed explanation of the clustering analysis


@dataclass
class DiversityReport:
    """Report schema for source diversity analysis."""

    geographic_diversity: str  # Geographic diversity of sources: "Low", "Medium", or "High"
    ideological_diversity: str  # Ideological diversity of sources: "Low", "Medium", or "High"
    type_diversity: str  # Diversity of source types (primary/secondary/tertiary): "Low", "Medium", or "High"
    explanation: str  # Detailed explanation of selection bias

@dataclass
class VerificationReport:
    """Report schema for claim verification against source content."""

    verification_score: float  # Score from 0 to 1 indicating how well the source verifies the claim (1 = very strong verification)
    explanation: str  # Detailed explanation of the verification analysis
    content_summary: str  # Brief summary of what the source actually says
