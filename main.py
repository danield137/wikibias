"""Main entry point for Wikipedia bias analysis."""

import argparse
import json
from typing import Optional
from dotenv import load_dotenv

from wikibias.llm import model_provider
from wikibias.wiki import get_text_and_refs
from wikibias.analyze import orchestrate_paragraph_analysis, generate_paragraph_summary, generate_page_summary

# Initialize global model getter
get_model = model_provider(local=True)


def analyze_wikipedia_page(title: str, max_paragraphs: Optional[int] = None) -> str:
    """Main analysis pipeline using the Orchestrator-Tool architecture.

    Args:
        title: Wikipedia page title
        max_paragraphs: Maximum number of paragraphs to analyze (None for all)

    Returns:
        JSON string containing the complete analysis
    """
    print(f"Analyzing Wikipedia page: {title}")

    # Fetch content with citations preserved
    paragraphs, refs = get_text_and_refs(title)
    print(f"Found {len(paragraphs)} paragraphs, {len(refs)} references")

    # Limit paragraphs if specified
    if max_paragraphs is not None and max_paragraphs > 0:
        paragraphs = paragraphs[:max_paragraphs]
        print(f"Limiting analysis to first {len(paragraphs)} paragraphs")

    # Use the article title as the topic
    article_topic = title.replace("_", " ")

    # Analyze each paragraph using the orchestrator
    paragraph_reports = []
    paragraph_summaries = []

    for i, paragraph in enumerate(paragraphs, 1):
        print(f"\nProcessing paragraph {i}/{len(paragraphs)}...")

        # Orchestrate the complete analysis for this paragraph
        report_card = orchestrate_paragraph_analysis(
            paragraph=paragraph, refs=refs, article_topic=article_topic, get_model=get_model
        )

        # Generate a summary for this paragraph
        summary = generate_paragraph_summary(report_card, get_model)

        paragraph_reports.append(report_card)
        paragraph_summaries.append(summary)

        print(
            f"  Paragraph {i} summary: Bias={summary.get('overall_bias_score', 'N/A')}/10, "
            f"Factuality={summary.get('overall_factuality_score', 'N/A')}/10"
        )

    # Generate page-level summary
    print("\nGenerating page-level summary...")
    page_summary = generate_page_summary(paragraph_summaries, get_model)

    # Compile final report
    final_report = {
        "article_title": title,
        "article_topic": article_topic,
        "total_paragraphs_analyzed": len(paragraphs),
        "paragraph_reports": paragraph_reports,
        "paragraph_summaries": paragraph_summaries,
        "page_summary": page_summary,
    }

    return json.dumps(final_report, indent=2)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze Wikipedia page for bias and factuality")
    parser.add_argument("title", type=str, help='Wikipedia page title (e.g., "Gaza_war")')
    parser.add_argument(
        "--max-paragraphs", type=int, default=None, help="Maximum number of paragraphs to analyze (default: all)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: print to console)")

    args = parser.parse_args()

    result = analyze_wikipedia_page(args.title, max_paragraphs=args.max_paragraphs)

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "=" * 80)
        print("Analysis Result:")
        print("=" * 80)
        print(result)


if __name__ == "__main__":
    main()
