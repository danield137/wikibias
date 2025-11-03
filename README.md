# WikiBias

A tool for analyzing Wikipedia articles to detect various forms of bias and verify factual accuracy through automated source checking.

**⚠️ Note: This is a prototype. Currently only supports analyzing 1 paragraph at a time.**

## What it does

WikiBias analyzes Wikipedia pages by:
- Detecting 15+ types of textual bias (loaded language, framing bias, missing context, etc.)
- Verifying claims against their cited sources by scraping and analyzing the actual content
- Evaluating source quality, diversity, and clustering
- Generating bias and factuality scores with detailed explanations

## Setup

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your environment:
```bash
cp .env.sample .env
```

Edit `.env` to set up your LLM provider:
- For local LLM: Set `LOCAL=True` and configure `API_BASE` (e.g., for LM Studio)
- For OpenAI: Set `LOCAL=False` and add your `OPENAI_API_KEY`

## Usage

Analyze a Wikipedia article:
```bash
python main.py "Stalin" --max-paragraphs 1
```

Options:
- `--max-paragraphs N`: Limit analysis to first N paragraphs (currently max 1)


## Output

The tool generates a comprehensive JSON report including:
- Paragraph-level bias findings and source analyses
- Overall bias score (0-10)
- Overall factuality score (0-10)
- Political alignment detection
- Detailed explanations for each finding