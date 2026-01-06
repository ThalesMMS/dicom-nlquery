#!/usr/bin/env python3
"""
Manual test script for dicom-nlquery.

Usage:
  python scripts/manual_search_cli.py "consulta em pt-br"
  python scripts/manual_search_cli.py --config config.yaml "consulta"

Prerequisites:
  - Ollama running with the model configured in config.yaml
  - DICOM node reachable from this machine
  - dicom-mcp available (uv pip install -e ../dicom-mcp)
"""

from __future__ import annotations

import argparse
import sys

from dicom_nlquery.config import load_config
from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.nl_parser import parse_nl_to_criteria


def _create_dicom_client(config):
    try:
        from dicom_mcp.dicom_client import DicomClient
    except ModuleNotFoundError:
        print("dicom-mcp nao encontrado. Instale com: uv pip install -e ../dicom-mcp")
        sys.exit(3)

    node = config.nodes[config.current_node]
    return DicomClient(
        host=node.host,
        port=node.port,
        calling_aet=config.calling_aet,
        called_aet=node.ae_title,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual dicom-nlquery search")
    parser.add_argument("query", help="Consulta em linguagem natural")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Arquivo de configuracao (default: config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config carregado: node={config.current_node}")

    criteria = parse_nl_to_criteria(args.query, config.llm)
    print("Criteria:")
    print(criteria.model_dump_json(indent=2))

    dicom_client = _create_dicom_client(config)
    result = execute_search(
        criteria,
        dicom_client,
        matching_config=config.matching,
        guardrails_config=config.guardrails,
    )

    print(f"Estudos encontrados: {len(result.accession_numbers)}")
    for accession in result.accession_numbers:
        print(accession)


if __name__ == "__main__":
    main()
