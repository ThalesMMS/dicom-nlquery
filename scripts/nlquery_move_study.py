#!/usr/bin/env python3
"""
Natural language DICOM query via dicom-nlquery + C-MOVE via dicom-mcp.

Example:
  python scripts/nlquery_move_study.py \
    "mulheres de 20 a 40 anos com cranio" \
    --host localhost --port 11112 --called-aet RADIANT \
    --calling-aet MCPSCU --destination-ae MONAI-DEPLOY \
    --date-range 20100101-20991231

Notes:
- Requires Ollama running locally (default: http://127.0.0.1:11434).
- RADIANT must allow the calling AET and know MONAI-DEPLOY as a move destination.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

from dicom_nlquery.config import load_config
from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.models import GuardrailsConfig, LLMConfig, MatchingConfig
from dicom_nlquery.nl_parser import parse_nl_to_criteria


def _load_dicom_mcp_client():
    try:
        from dicom_mcp.dicom_client import DicomClient

        return DicomClient
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        src_dir = repo_root / "dicom-mcp" / "src" / "dicom_mcp"
        if not src_dir.exists():
            raise RuntimeError(
                f"dicom-mcp src not found at {src_dir}. Install it or adjust the path."
            )

        pkg = sys.modules.get("dicom_mcp")
        if pkg is None:
            pkg = types.ModuleType("dicom_mcp")
            pkg.__path__ = [str(src_dir)]
            sys.modules["dicom_mcp"] = pkg

        for name in ("attributes", "dicom_client"):
            module_name = f"dicom_mcp.{name}"
            if module_name in sys.modules:
                continue
            path = src_dir / f"{name}.py"
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            assert spec and spec.loader
            spec.loader.exec_module(module)

        return sys.modules["dicom_mcp.dicom_client"].DicomClient


def _load_optional_config(path: str | None):
    if path:
        return load_config(path)
    default_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if default_path.exists():
        return load_config(default_path)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NL query via dicom-nlquery + C-MOVE via dicom-mcp",
    )
    parser.add_argument("query", help="Natural language query (PT-BR)")
    parser.add_argument("--config", default=None, help="dicom-nlquery config.yaml")
    parser.add_argument("--host", default="localhost", help="DICOM SCP host")
    parser.add_argument("--port", type=int, default=11112, help="DICOM SCP port")
    parser.add_argument("--called-aet", default="RADIANT", help="Called AE title")
    parser.add_argument("--calling-aet", default="MCPSCU", help="Calling AE title")
    parser.add_argument(
        "--destination-ae",
        default="MONAI-DEPLOY",
        help="Destination AE title for C-MOVE",
    )
    parser.add_argument(
        "--date-range",
        default=None,
        help="Date range YYYYMMDD-YYYYMMDD (optional)",
    )
    parser.add_argument(
        "--max-studies",
        type=int,
        default=None,
        help="Max studies to scan (optional)",
    )
    parser.add_argument("--unlimited", action="store_true", help="Disable guardrails")
    parser.add_argument(
        "--move-all",
        action="store_true",
        help="Move all matched accessions (default: first only)",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Override LLM base URL (default: config or http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override LLM model (default: config or llama3.2:latest)",
    )
    args = parser.parse_args()

    config = _load_optional_config(args.config)
    if config:
        llm_config = config.llm
        guardrails_config = config.guardrails
        matching_config = config.matching
    else:
        llm_config = LLMConfig(
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            model="llama3.2:latest",
            temperature=0,
            timeout=60,
        )
        guardrails_config = GuardrailsConfig()
        matching_config = MatchingConfig()

    if args.llm_base_url:
        llm_config = llm_config.model_copy(update={"base_url": args.llm_base_url})
    if args.llm_model:
        llm_config = llm_config.model_copy(update={"model": args.llm_model})

    try:
        criteria = parse_nl_to_criteria(args.query, llm_config)
    except Exception as exc:
        print(f"Erro ao parsear a consulta: {exc}")
        return 3

    try:
        DicomClient = _load_dicom_mcp_client()
        client = DicomClient(
            host=args.host,
            port=args.port,
            calling_aet=args.calling_aet,
            called_aet=args.called_aet,
        )
    except Exception as exc:
        print(f"Erro ao inicializar DICOM client: {exc}")
        return 2

    try:
        result = execute_search(
            criteria,
            client,
            matching_config=matching_config,
            guardrails_config=guardrails_config,
            date_range=args.date_range,
            max_studies=args.max_studies,
            unlimited=args.unlimited,
        )
    except Exception as exc:
        print(f"Erro na busca DICOM: {exc}")
        return 2

    accessions = list(dict.fromkeys(result.accession_numbers))
    if not accessions:
        print("Nenhum estudo encontrado para mover.")
        return 1

    targets = accessions if args.move_all else accessions[:1]
    print(
        f"Encontrados {len(accessions)} accession(s); movendo {len(targets)} estudo(s) para {args.destination_ae}."
    )

    successes = 0
    for accession in targets:
        studies = client.query_study(
            accession_number=accession,
            additional_attrs=["StudyInstanceUID"],
        )
        if not studies:
            print(f"StudyInstanceUID nao encontrado para {accession}")
            continue
        study_uid = studies[0].get("StudyInstanceUID")
        if not study_uid:
            print(f"StudyInstanceUID ausente para {accession}")
            continue

        move_result = client.move_study(
            destination_ae=args.destination_ae,
            study_instance_uid=study_uid,
        )
        print(f"Move {accession}: {move_result}")
        if move_result.get("success"):
            successes += 1

    if successes == 0:
        print("Nenhum estudo foi movido com sucesso.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
