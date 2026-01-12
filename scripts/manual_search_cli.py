#!/usr/bin/env python3
"""
Manual test script for dicom-nlquery.

Usage:
  python scripts/manual_search_cli.py "query in en-us"
  python scripts/manual_search_cli.py --config config.yaml "query"

Prerequisites:
  - Ollama running with the model configured via llm_path (configs/llm-test.yaml)
  - dicom-mcp available (uv pip install -e ../dicom-mcp)
"""

from __future__ import annotations

import argparse
from dicom_nlquery.config import load_config
from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.nl_parser import parse_nl_to_criteria


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual dicom-nlquery search")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print("Config loaded: dicom-mcp")
    if config.mcp is None:
        print("Error: mcp.config_path not configured in config.yaml")
        return

    criteria = parse_nl_to_criteria(args.query, config.llm, strict_evidence=True)
    print("Criteria:")
    print(criteria.model_dump_json(indent=2))

    result = execute_search(
        criteria,
        mcp_config=config.mcp,
        guardrails_config=config.guardrails,
        search_pipeline_config=config.search_pipeline,
        lexicon_config=config.lexicon,
        rag_config=config.rag,
        ranking_config=config.ranking,
        rag_query=args.query,
    )

    print(f"Studies found: {len(result.accession_numbers)}")
    for accession in result.accession_numbers:
        print(accession)


if __name__ == "__main__":
    main()
