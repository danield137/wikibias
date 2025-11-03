from typing import List, Callable, Dict, Any
import dataclasses
import json
from .schemas import BiasFinding, SourceAnalysis
from .llm import extract_json_from_result, create_agent
from .text_scanner import TEXT_SCANNER_TOOLS
from .source_analyzer import SOURCE_ANALYZER_TOOLS


def parse_paragraph_into_claims(paragraph: str, get_model: Callable) -> List[str]:
    """Parse a paragraph into individual claims or sentences.

    Args:
        paragraph: The paragraph text to parse
        get_model: Function to get the LLM model

    Returns:
        List of claim strings
    """
    agent = create_agent(
        name="ClaimParser",
        instructions="""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. You are a paragraph parser. Given a paragraph, extract individual claims or sentences.
        Each claim should be a standalone statement that can be analyzed independently. 
        INCLUDE citation markers (e.g., [1], [2]) as they appear in the text !!
        
        CRITICAL: You MUST ALWAYS return ONLY a valid JSON object. Never return plain text, explanations, or any other format.
        If you cannot parse the paragraph or have no results, return {"claims": []}.
        Do not include any text before or after the JSON object.
        
        IMPORTANT: Escape all double quotes in string values as \\"
        
        Output format:
        {
          "claims": ["claim 1", "claim 2", "claim 3", ...]
        }
        
        Example:
        Input: "The war began on October 7, 2023 [1]. Hamas launched a surprise attack [3][4][5]. Over 1,000 people were killed [2]."
        Output: {
          "claims": [
            "The war began on October 7, 2023. [1]",
            "Hamas launched a surprise attack. [3][4][5]",
            "Over 1,000 people were killed. [2]"
          ]
        }
        """,
        get_model=get_model,
    )

    result = agent.run(f"Parse this paragraph into claims:\n\n{paragraph}")
    try:
        data = extract_json_from_result(result)
        return data.get("claims", [])
    except Exception as e:
        print(f"  Warning: Failed to parse paragraph into claims: {str(e)[:100]}")
        # Fallback: split by sentence
        import re

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        return [s.strip() for s in sentences if s.strip()]


def extract_citation_markers(claim: str) -> List[int|str]:
    """Extract citation markers from a claim (e.g., [1], [2], [o]).

    Args:
        claim: The claim text

    Returns:
        List of citation indices
    """
    import re

    pattern = r"\[([a-zA-Z\d]+)\]"
    matches = re.findall(pattern, claim)
    return [m for m in matches]


def run_text_scanners(paragraph: str, article_topic: str, get_model: Callable) -> List[BiasFinding]:
    """Run all text-scanning tools on the paragraph.

    Args:
        paragraph: The paragraph text to analyze
        article_topic: The article topic for context (used by narrative_framing)
        get_model: Function to get the LLM model

    Returns:
        List of all BiasFinding objects from all tools
    """
    all_findings = []

    # Run all text-scanning tools
    for tool_name, tool_func in TEXT_SCANNER_TOOLS.items():
        if tool_name == "analyze_narrative_framing":
            # This tool requires article_topic parameter
            findings = tool_func(paragraph, article_topic, get_model)
        else:
            findings = tool_func(paragraph, get_model)

        all_findings.extend(findings)

    return all_findings

def run_source_analyzers(
    claim: str, citation_indices: List[int|str], refs: List[Dict], get_model: Callable
) -> List[SourceAnalysis]:
    """Run source analysis tools on citations for a claim.

    Args:
        claim: The claim text
        citation_indices: List of citation indices in the claim
        refs: List of reference dicts from Wikipedia
        get_model: Function to get the LLM model

    Returns:
        List of SourceAnalysis objects
    """
    all_analyses = []

    # Get citation details for this claim
    claim_citations = []
    for idx in citation_indices:
        matching_ref = next((r for r in refs if r["key"] == idx), None)
        if matching_ref:
            claim_citations.append(matching_ref)

    if not claim_citations:
        return all_analyses

    # Run claim verification by scraping actual source content
    for citation in claim_citations:
        # TODO: notes don't necessarily have URLs. Proper handling needed. potentially injecting notes in previous step as part of context for textual bias analysis
        if citation.get("url"):
            verification_analysis = SOURCE_ANALYZER_TOOLS["verify_claim_against_source"](
                claim_text=claim,
                source_url=citation["url"],
                citation_index=citation["key"],
                get_model=get_model,
            )
            all_analyses.append(verification_analysis)
            
            # Check if this is a weak source (verification score < 0.5)
            verification_score = verification_analysis.report.get("verification_score", 0.0)
            if verification_score < 0.5:
                print(f"        ⚠️  Weak source detected (score: {verification_score:.2f})")
    # Run source integrity analysis on each citation
    for citation in claim_citations:
        if citation.get("url"):
            integrity_analysis = SOURCE_ANALYZER_TOOLS["analyze_source_integrity"](
                claim_text=claim,
                source_url=citation["url"],
                source_description=citation.get("text", ""),
                get_model=get_model,
            )
            all_analyses.append(integrity_analysis)

    # Run clustering analysis if multiple citations
    if len(claim_citations) > 1:
        source_descriptions = [c.get("text", "") for c in claim_citations]
        clustering_analysis = SOURCE_ANALYZER_TOOLS["analyze_citation_clustering"](
            claim_text=claim, source_list=source_descriptions, get_model=get_model
        )
        all_analyses.append(clustering_analysis)

    # Run diversity analysis if we have citations with URLs
    citations_with_urls = [
        {"description": c.get("text", ""), "url": c.get("url", "")} for c in claim_citations if c.get("url")
    ]
    if citations_with_urls:
        diversity_analysis = SOURCE_ANALYZER_TOOLS["analyze_source_diversity"](
            source_list=citations_with_urls, get_model=get_model
        )
        all_analyses.append(diversity_analysis)

    return all_analyses


def orchestrate_paragraph_analysis(
    paragraph: str, refs: List[Dict], article_topic: str, get_model: Callable
) -> Dict[str, Any]:
    """Orchestrate the complete analysis of a paragraph.

    This is the main orchestrator that:
    1. Runs text-scanning tools on the entire paragraph
    2. Parses the paragraph into claims for source analysis
    3. Runs source analysis tools on citations for each claim
    4. Aggregates all findings into a Bias Report Card

    Args:
        paragraph: The paragraph text
        refs: List of reference dicts from Wikipedia
        article_topic: The article topic for context
        get_model: Function to get the LLM model

    Returns:
        Dict containing the complete bias report card
    """
    print(f"  Orchestrating analysis...")

    # Step 0: Escape double quotes in paragraph
    paragraph = paragraph.replace('"', '\\"')

    # Step 1: Run text-scanning tools on the entire paragraph
    print(f"  Running text scanners on paragraph...")
    text_findings = run_text_scanners(paragraph, article_topic, get_model)
    print(f"  Found {len(text_findings)} text bias signals")

    # Step 2: Parse paragraph into claims for source analysis
    claims = parse_paragraph_into_claims(paragraph, get_model)
    print(f"  Parsed into {len(claims)} claims for source analysis")

    # Step 3: Analyze sources for each claim
    claim_reports = []

    for i, claim in enumerate(claims, 1):
        print(f"    Analyzing sources for claim {i}/{len(claims)}...")

        # Extract citation markers
        citation_indices = extract_citation_markers(claim)

        # Run source analysis tools (only if citations exist)
        source_analyses = []
        if citation_indices:
            source_analyses = run_source_analyzers(claim, citation_indices, refs, get_model)
            print(f"      Completed {len(source_analyses)} source analyses")

        claim_reports.append(
            {
                "claim": claim,
                "citation_indices": citation_indices,
                "source_analyses": [
                    {"source_id": sa.source_id, "analysis_type": sa.analysis_type, "report": sa.report}
                    for sa in source_analyses
                ],
            }
        )

    # Step 4: Aggregate into Bias Report Card
    return {
        "paragraph": paragraph,
        "article_topic": article_topic,
        "text_findings": [dataclasses.asdict(f) for f in text_findings],
        "claim_reports": claim_reports,
        "summary": {
            "total_claims": len(claims),
            "total_text_findings": len(text_findings),
            "total_source_analyses": sum(len(cr["source_analyses"]) for cr in claim_reports),
        },
    }


def _create_lean_report(report_card: Dict[str, Any]) -> Dict[str, Any]:
    """Create a lean version of the report card by removing verbose fields.
    
    This is used when the full report is too large for the LLM's context window.
    
    Args:
        report_card: The full bias report card
        
    Returns:
        A lean version with only essential fields
    """
    lean_card = {
        "paragraph": report_card.get("paragraph", ""),
        "article_topic": report_card.get("article_topic", ""),
        "text_findings": [],
        "claim_reports": [],
        "summary": report_card.get("summary", {}),
    }
    
    # For text findings, only keep kind, strength, and explanation (exclude text and span)
    for finding in report_card.get("text_findings", []):
        lean_card["text_findings"].append({
            "kind": finding.get("kind", ""),
            "strength": finding.get("strength", ""),
            "explanation": finding.get("explanation", ""),
        })
    
    # For claim reports, keep the structure but simplify source analyses
    for claim_report in report_card.get("claim_reports", []):
        lean_claim = {
            "claim": claim_report.get("claim", ""),
            "citation_indices": claim_report.get("citation_indices", []),
            "source_analyses": [],
        }
        
        # Simplify source analyses by removing verbose fields
        for analysis in claim_report.get("source_analyses", []):
            lean_analysis = {
                "source_id": analysis.get("source_id", ""),
                "analysis_type": analysis.get("analysis_type", ""),
                "report": {},
            }
            
            # Keep only summary fields from report, exclude full text content
            report = analysis.get("report", {})
            for key, value in report.items():
                # Skip fields that might contain large text content
                if key not in ["source_text", "full_text", "content", "raw_content"]:
                    lean_analysis["report"][key] = value
            
            lean_claim["source_analyses"].append(lean_analysis)
        
        lean_card["claim_reports"].append(lean_claim)
    
    return lean_card

def generate_paragraph_summary(report_card: Dict[str, Any], get_model: Callable) -> Dict[str, Any]:
    """Generate a human-readable summary of the paragraph analysis.

    Args:
        report_card: The bias report card from orchestrate_paragraph_analysis
        get_model: Function to get the LLM model

    Returns:
        Dict with summary scores and explanations
    """
    agent = create_agent(
        name="ParagraphSummarizer",
        instructions="""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. You are a bias analysis summarizer. Given a detailed bias report card,
        provide a concise summary with overall scores.
        
        CRITICAL: You MUST ALWAYS return ONLY a valid JSON object. Never return plain text, explanations, or any other format.
        If you cannot analyze the report or have no results, return a JSON object with default values.
        Do not include any text before or after the JSON object.
        
        IMPORTANT: Escape all double quotes in string values as \\"
        
        Output format:
        {
          "overall_bias_score": <0-10>,
          "overall_factuality_score": <0-10>,
          "political_leaning": "<Left([-1,0)|Right([0,1]|Center(≈0)>",
          "representative_example": "a direct quote from the text that best exemplifies the bias found",
          "key_issues": ["issue 1", "issue 2", ...],
          "summary": "brief summary of findings"
        }
        
        For political_leaning, negative scores indicate Left-leaning, positive scores indicate Right-leaning, and near-zero scores indicate Center.
        
        For representative_example:
        - Select a direct quote from the paragraph that best demonstrates the most significant bias
        - This should be a concrete example that readers can immediately understand
        - If no significant bias is found, you may use an empty string ""
        """,
        get_model=get_model,
    )

    # Try with the full report first
    prompt = f"Analyze this bias report card and provide a summary:\n\n{json.dumps(report_card, indent=2)}"
    
    try:
        result = agent.run(prompt)
        return extract_json_from_result(result)
    except Exception as e:
        error_msg = str(e)
        # Check if it's a context length error
        if "context length" in error_msg.lower() or "400" in error_msg:
            print(f"  Warning: Report too large for context window, using lean version...")
            
            # Create a lean version and retry
            lean_report = _create_lean_report(report_card)
            lean_prompt = f"Analyze this bias report card and provide a summary:\n\n{json.dumps(lean_report, indent=2)}"
            
            try:
                result = agent.run(lean_prompt)
                return extract_json_from_result(result)
            except Exception as retry_error:
                print(f"  Warning: Failed to generate summary even with lean report: {str(retry_error)[:100]}")
                return {
                    "overall_bias_score": 5,
                    "overall_factuality_score": 5,
                    "key_issues": [],
                    "summary": "Error generating summary - report too large",
                }
        else:
            print(f"  Warning: Failed to generate summary: {error_msg[:100]}")
            return {
                "overall_bias_score": 5,
                "overall_factuality_score": 5,
                "key_issues": [],
                "summary": "Error generating summary",
            }


def generate_page_summary(paragraph_summaries: List[Dict[str, Any]], get_model: Callable) -> Dict[str, Any]:
    """Generate an overall summary for the entire page.

    Args:
        paragraph_summaries: List of paragraph summary dicts
        get_model: Function to get the LLM model

    Returns:
        Dict with overall page summary
    """
    agent = create_agent(
        name="PageSummarizer",
        instructions="""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. You are a page-level bias summarizer. Given summaries of multiple paragraphs,
        provide a concise overall assessment of the page's bias and factuality.
        
        CRITICAL: You MUST ALWAYS return ONLY a valid JSON object. Never return plain text, explanations, or any other format.
        If you cannot analyze the summaries or have no results, return a JSON object with default values.
        Do not include any text before or after the JSON object.
        
        IMPORTANT: Escape all double quotes in string values as \\"
        
        Output format:
        {
          "overall_bias_score": <0-10>,
          "overall_factuality_score": <0-10>,
          "overall_political_leaning": "<Left[-1,0]|Right[0,1]|Center(≈0)>",
          "representative_examples": ["example 1", "example 2", "example 3"],
          "summary": "comprehensive summary of page bias and factuality - make this intriguing and click-baity while remaining factual"
        }
        
        For overall_political_leaning:
        - Synthesize the political_leaning from all paragraph summaries
        - Use "Left", "Right", "Center". Indicated strength with brackets: Left[-1,0], Right[0,1], Center(≈0)
        
        For representative_examples:
        - Select the 3-5 most compelling examples from all paragraph summaries
        - These should be the most striking instances of bias found
        - Include direct quotes that demonstrate the bias clearly
        
        For summary:
        - Write an engaging, intriguing summary that captures attention
        - Be factual but make it compelling - think NY Times headline style
        - Highlight the most surprising or significant findings
        - Keep it concise but impactful
        - Use formal or journalistic tone, don't sound like clickbait
        """,
        get_model=get_model,
    )

    prompt = f"Summarize these paragraph analyses:\n\n{json.dumps(paragraph_summaries, indent=2)}"
    result = agent.run(prompt)

    try:
        return extract_json_from_result(result)
    except Exception as e:
        print(f"  Warning: Failed to generate page summary: {str(e)[:100]}")
        return {
            "overall_bias_score": 5,
            "overall_factuality_score": 5,
            "summary": "Error generating page summary",
        }
