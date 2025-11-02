from typing import Callable
import json
from .schemas import SourceAnalysis, IntegrityReport, ClusteringReport, DiversityReport
from .llm import extract_json_from_result, create_agent


def analyze_source_integrity(
    claim_text: str, source_url: str, source_description: str, get_model: Callable
) -> SourceAnalysis:
    """Analyze a single source against a single claim for reliability and bias.

    Args:
        claim_text: The claim being verified
        source_url: URL of the source
        source_description: Description of the source
        get_model: Function to get the LLM model

    Returns:
        SourceAnalysis object with analysis_type='integrity'
    """
    agent = create_agent(
        name="SourceIntegrityAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the provided source against the claim. Return a SourceAnalysis object with 
        'analysis_type': 'integrity' and a report containing:
        - 'source_reliability' (0-1): How reliable is this source?
        - 'source_bias_score' (-1 to 1): Ideological bias (-1=left, 0=neutral, 1=right)
        - 'verification_strength' ('Full', 'Partial', or 'None'): How well does it verify the claim?
        - 'explanation': Detailed explanation
        
        IMPORTANT: Escape all double quotes in string values as \\"
        
        Output ONLY valid JSON in this format:
        {{
          "source_id": "source description",
          "analysis_type": "integrity",
          "report": {{
            "source_reliability": <0.0-1.0>,
            "source_bias_score": <-1.0 to 1.0>,
            "verification_strength": "Full" or "Partial" or "None",
            "explanation": "detailed explanation"
          }}
        }}
        """,
        get_model=get_model,
    )

    prompt = f"""Claim: {claim_text}
Source URL: {source_url}
Source Description: {source_description}"""

    result = agent.run(prompt)
    try:
        data = extract_json_from_result(result)
        return SourceAnalysis(
            source_id=data.get("source_id", source_description),
            analysis_type="integrity",
            report=data.get("report", {}),
        )
    except Exception as e:
        print(f"  Warning: Failed to parse source_integrity analysis: {str(e)[:100]}")
        return SourceAnalysis(
            source_id=source_description,
            analysis_type="integrity",
            report={
                "source_reliability": 0.5,
                "source_bias_score": 0.0,
                "verification_strength": "None",
                "explanation": f"Error parsing analysis: {str(e)[:100]}",
            },
        )


def analyze_citation_clustering(claim_text: str, source_list: list[str], get_model: Callable) -> SourceAnalysis:
    """Analyze if citations cluster around a single original source.

    Args:
        claim_text: The claim being cited
        source_list: list of source descriptions
        get_model: Function to get the LLM model

    Returns:
        SourceAnalysis object with analysis_type='clustering'
    """
    agent = create_agent(
        name="CitationClusteringAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze this list of sources for a single claim. Do they 'cluster' around one original source?
        
        Return a SourceAnalysis object with 'analysis_type': 'clustering' and a report containing:
        - 'is_clustered' (bool): Do they cluster?
        - 'independent_sources' (int): Number of truly independent sources
        - 'total_citations' (int): Total citations analyzed
        - 'original_source' (string): The original/primary source if clustered
        - 'explanation': Detailed explanation
        
        IMPORTANT: Escape all double quotes in string values as \\"
        
        Output ONLY valid JSON in this format:
        {{
          "source_id": "clustering analysis",
          "analysis_type": "clustering",
          "report": {{
            "is_clustered": true or false,
            "independent_sources": <int>,
            "total_citations": <int>,
            "original_source": "source name or empty string",
            "explanation": "detailed explanation"
          }}
        }}
        """,
        get_model=get_model,
    )

    prompt = f"""Claim: {claim_text}
Sources: {json.dumps(source_list, indent=2)}"""

    result = agent.run(prompt)
    try:
        data = extract_json_from_result(result)
        return SourceAnalysis(
            source_id=data.get("source_id", "clustering analysis"),
            analysis_type="clustering",
            report=data.get("report", {}),
        )
    except Exception as e:
        print(f"  Warning: Failed to parse citation_clustering analysis: {str(e)[:100]}")
        return SourceAnalysis(
            source_id="clustering analysis",
            analysis_type="clustering",
            report={
                "is_clustered": False,
                "independent_sources": len(source_list),
                "total_citations": len(source_list),
                "original_source": "",
                "explanation": f"Error parsing analysis: {str(e)[:100]}",
            },
        )


def analyze_source_diversity(source_list: list[dict[str, str]], get_model: Callable) -> SourceAnalysis:
    """Analyze diversity of sources for selection bias.

    Args:
        source_list: list of dicts with 'description' and 'url' keys
        get_model: Function to get the LLM model

    Returns:
        SourceAnalysis object with analysis_type='diversity'
    """
    agent = create_agent(
        name="SourceDiversityAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the diversity of this source list. Return a SourceAnalysis object with 
        'analysis_type': 'diversity' and a report containing:
        - 'geographic_diversity' ('Low', 'Medium', or 'High')
        - 'ideological_diversity' ('Low', 'Medium', or 'High')
        - 'type_diversity' ('Low', 'Medium', or 'High'): primary/secondary/tertiary mix
        - 'explanation': Explanation of selection bias
        
        IMPORTANT: Escape all double quotes in string values as \\"
        
        Output ONLY valid JSON in this format:
        {{
          "source_id": "diversity analysis",
          "analysis_type": "diversity",
          "report": {{
            "geographic_diversity": "Low" or "Medium" or "High",
            "ideological_diversity": "Low" or "Medium" or "High",
            "type_diversity": "Low" or "Medium" or "High",
            "explanation": "detailed explanation of selection bias"
          }}
        }}
        """,
        get_model=get_model,
    )

    prompt = f"Sources: {json.dumps(source_list, indent=2)}"

    result = agent.run(prompt)
    try:
        data = extract_json_from_result(result)
        return SourceAnalysis(
            source_id=data.get("source_id", "diversity analysis"),
            analysis_type="diversity",
            report=data.get("report", {}),
        )
    except Exception as e:
        print(f"  Warning: Failed to parse source_diversity analysis: {str(e)[:100]}")
        return SourceAnalysis(
            source_id="diversity analysis",
            analysis_type="diversity",
            report={
                "geographic_diversity": "Medium",
                "ideological_diversity": "Medium",
                "type_diversity": "Medium",
                "explanation": f"Error parsing analysis: {str(e)[:100]}",
            },
        )


def verify_claim_against_source(
    claim_text: str, source_url: str, citation_index: int, get_model: Callable
) -> SourceAnalysis:
    """Verify a claim by scraping and analyzing the actual source content.

    Args:
        claim_text: The claim to verify
        source_url: URL of the source to scrape
        citation_index: The citation index number
        get_model: Function to get the LLM model

    Returns:
        SourceAnalysis object with analysis_type='verification'
    """
    from .scrape import scrape_url_content, chunk_text_for_llm
    
    try:
        # Scrape the content from the URL
        print(f"        Scraping content from {source_url}...")
        try:
            paragraphs = scrape_url_content(source_url)
        except Exception as scrape_error:
            # If we cannot scrape the URL, treat it as a bad source
            print(f"        Failed to scrape URL: {str(scrape_error)[:100]}")
            return SourceAnalysis(
                source_id=f"citation [{citation_index}]: {source_url}",
                analysis_type="verification",
                report={
                    "verification_score": 0.0,
                    "explanation": f"Failed to access or scrape source: {str(scrape_error)[:200]}",
                    "content_summary": "Source inaccessible - treated as bad source",
                },
            )
        
        if not paragraphs:
            # No content extracted - treat as bad source
            return SourceAnalysis(
                source_id=f"citation [{citation_index}]: {source_url}",
                analysis_type="verification",
                report={
                    "verification_score": 0.0,
                    "explanation": "Could not extract meaningful content from the source URL",
                    "content_summary": "No content found - treated as bad source",
                },
            )
        
        # Chunk the content for LLM processing
        chunks = chunk_text_for_llm(paragraphs, max_chars=8000)
        print(f"        Extracted {len(paragraphs)} paragraphs, chunked into {len(chunks)} segments")
        
        # Analyze each chunk and aggregate results
        chunk_scores = []
        chunk_explanations = []
        
        for i, chunk in enumerate(chunks):
            print(f"        Analyzing chunk {i+1}/{len(chunks)}...")
            
            agent = create_agent(
                name="ClaimVerificationAnalyzer",
                instructions=f"""
                You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze whether the following source text verifies the given claim.
                
                Return a verification score from 0.0 to 1.0 where:
                - 1.0 = The source strongly verifies the claim with clear evidence
                - 0.7-0.9 = The source supports the claim with good evidence
                - 0.5-0.6 = The source partially supports the claim
                - 0.3-0.4 = The source mentions the topic but doesn't clearly verify the claim
                - 0.0-0.2 = The source contradicts the claim or doesn't mention it
                
                Also provide:
                - A brief summary of what the source actually says
                - A detailed explanation of how well it verifies the claim
                
                IMPORTANT: Escape all double quotes in string values as \\"
                
                Output ONLY valid JSON in this format:
                {{
                  "verification_score": <0.0-1.0>,
                  "content_summary": "brief summary of source content",
                  "explanation": "detailed explanation of verification analysis"
                }}
                """,
                get_model=get_model,
            )
            
            prompt = f"""Claim: {claim_text}

Source text from citation [{citation_index}]: {source_url}

{chunk}"""
            
            result = agent.run(prompt)
            try:
                data = extract_json_from_result(result)
                chunk_scores.append(data.get("verification_score", 0.0))
                chunk_explanations.append({
                    "chunk": i + 1,
                    "score": data.get("verification_score", 0.0),
                    "summary": data.get("content_summary", ""),
                    "explanation": data.get("explanation", ""),
                })
            except Exception as e:
                print(f"        Warning: Failed to parse chunk {i+1} verification: {str(e)[:100]}")
                chunk_scores.append(0.0)
        
        # Aggregate the results - use the maximum score (most supportive chunk)
        final_score = max(chunk_scores) if chunk_scores else 0.0
        
        # Find the best matching chunk explanation
        best_chunk = max(chunk_explanations, key=lambda x: x["score"]) if chunk_explanations else None
        
        if best_chunk:
            content_summary = best_chunk["summary"]
            explanation = f"Analyzed {len(chunks)} content segment(s). Best match (chunk {best_chunk['chunk']}, score: {best_chunk['score']:.2f}): {best_chunk['explanation']}"
        else:
            content_summary = "Analysis failed"
            explanation = "Could not analyze source content"
        
        return SourceAnalysis(
            source_id=f"citation [{citation_index}]: {source_url}",
            analysis_type="verification",
            report={
                "verification_score": final_score,
                "explanation": explanation,
                "content_summary": content_summary,
            },
        )
        
    except Exception as e:
        print(f"        Warning: Failed to verify claim against source: {str(e)[:200]}")
        return SourceAnalysis(
            source_id=f"citation [{citation_index}]: {source_url}",
            analysis_type="verification",
            report={
                "verification_score": 0.0,
                "explanation": f"Error during verification: {str(e)[:200]}",
                "content_summary": "Verification failed due to error",
            },
        )

# Registry of all source analysis tools
SOURCE_ANALYZER_TOOLS = {
    "analyze_source_integrity": analyze_source_integrity,
    "analyze_citation_clustering": analyze_citation_clustering,
    "analyze_source_diversity": analyze_source_diversity,
    "verify_claim_against_source": verify_claim_against_source,
}
