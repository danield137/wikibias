import ast
import logging
from typing import Any, Callable
from smolagents import OpenAIServerModel, ToolCallingAgent, LogLevel
import json

# Bias types for content analysis (paragraphs, pages)
CONTENT_BIAS_TYPES = [
    "Ideological / Political Bias",
    "Emotional / Sentiment Bias",
    "Selection / Omission Bias",
    "Framing Bias",
    "Confirmation / Cognitive Bias",
    "Geographic / Cultural Bias",
    "Commercial / Attention Bias",
    "Temporal / Recency Bias",
    "Visual / Imagery Bias",
    "Statistical / Data Bias",
    "Source Bias",
    "Narrative Bias",
]

# Bias types for source analysis
SOURCE_BIAS_TYPES = [
    "Ideological Alignment Bias",
    "Audience Confirmation Bias",
    "National / Cultural Bias",
    "Commercial / Ownership Bias",
    "Sensationalism / Attention Bias",
    "Access / Source Bias",
    "Elitism / Class Bias",
    "Status Quo / Institutional Bias",
    "Reputational / Brand Bias",
    "Geopolitical Bias",
    "Editorial Agenda Bias",
    "Platform Algorithmic Bias",
]


def model_provider(local=False):
    """Factory function that returns a model getter."""
    if local:
        model = OpenAIServerModel(
            model_id="openai/gpt-oss-20b",  #"google/gemma-3-12b",
            api_base="http://localhost:1234/v1",
            api_key="not-needed",
        )
    else:
        model = OpenAIServerModel(model_id="gpt-4o")

    def get_model():
        return model

    return get_model


def create_agent(name: str, instructions: str, get_model: Callable, tools: list | None = None) -> ToolCallingAgent:
    """Factory method to create a ToolCallingAgent with consistent settings.

    Args:
        name: Name of the agent
        instructions: Instructions for the agent
        get_model: Function that returns the model
        tools: List of tools for the agent (defaults to empty list)
        verbosity_level: Verbosity level for the agent (default: "critical")

    Returns:
        Configured ToolCallingAgent instance
    """
    if tools is None:
        tools = []

    return ToolCallingAgent(
        name=name,
        instructions=instructions,
        model=get_model(),
        tools=tools,
        verbosity_level=LogLevel.OFF,
        provide_run_summary=False,
    )

import json_repair


def load_messy_json(messy_json_str: str) -> dict:
    json_str = json_repair.repair_json(messy_json_str)
    return json.loads(json_str)

def extract_json_from_result(result_str: Any) -> dict:
    """Extract JSON from agent result string.

    The agent result may include extra text, so we try to find and parse just the JSON part.
    If the result is a plain string without JSON markers, return an empty dict (no bias found).
    """
    result_str = str(result_str).strip()

    # Check if the string contains JSON markers - if not, assume no bias found
    if '{' not in result_str and '[' not in result_str:
        print(f"  Warning: LLM returned non-JSON string, assuming no bias found: {result_str[:100]}")
        return {}

    # Try to parse the whole string first
    try:
        return load_messy_json(result_str)
    except:
        pass

    # Try to find JSON between curly braces
    start = result_str.find("{")
    end = result_str.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return load_messy_json(result_str[start : end + 1])
        except:
            pass

    # If all else fails, raise an error with the original string
    raise ValueError(f"Could not extract valid JSON from result: {result_str[:500]}")
