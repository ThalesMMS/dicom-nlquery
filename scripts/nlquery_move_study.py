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
import logging
import sys
import types
import warnings
from pathlib import Path

from dicom_nlquery.config import load_config
from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.models import GuardrailsConfig, LLMConfig, MatchingConfig
from dicom_nlquery.nl_parser import parse_nl_to_criteria
from dicom_nlquery.logging_config import JsonFormatter


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


def _setup_logging(verbose: bool, log_file: str | None) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(level)

    if log_file:
        handler = logging.FileHandler(log_file, mode="w")
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)

    return logging.getLogger("dicom_nlquery")


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
        "--destination-host",
        default=None,
        help="Destination host for post-move C-FIND verification",
    )
    parser.add_argument(
        "--destination-port",
        type=int,
        default=None,
        help="Destination port for post-move C-FIND verification",
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
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    parser.add_argument("--log-file", default=None, help="Write JSON logs to file")
    parser.add_argument(
        "--pydicom-warnings",
        action="store_true",
        help="Show pydicom validation warnings on stderr",
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

    if not args.pydicom_warnings:
        warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.valuerep")
        try:
            from pydicom import config as pydicom_config
        except Exception:
            pydicom_config = None
        if pydicom_config is not None:
            pydicom_config.settings.reading_validation_mode = pydicom_config.IGNORE

    logger = _setup_logging(args.verbose, args.log_file)
    logger.info(
        "Starting NL query move",
        extra={
            "extra_data": {
                "host": args.host,
                "port": args.port,
                "called_aet": args.called_aet,
                "calling_aet": args.calling_aet,
                "destination_ae": args.destination_ae,
                "destination_host": args.destination_host,
                "destination_port": args.destination_port,
                "date_range": args.date_range,
                "max_studies": args.max_studies,
                "unlimited": args.unlimited,
                "query_length": len(args.query),
            }
        },
    )

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
    logger.info("Criteria parsed", extra={"extra_data": criteria.model_dump()})

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
    destination_client = None
    if args.destination_host and args.destination_port:
        try:
            destination_client = DicomClient(
                host=args.destination_host,
                port=args.destination_port,
                calling_aet=args.calling_aet,
                called_aet=args.destination_ae,
            )
        except Exception as exc:
            print(f"Erro ao inicializar destino para verificacao: {exc}")
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
    logger.info(
        "Search completed",
        extra={
            "extra_data": {
                "accession_count": len(result.accession_numbers),
                "stats": result.stats.model_dump(),
            }
        },
    )

    accessions = list(dict.fromkeys(result.accession_numbers))
    if not accessions:
        print("Nenhum estudo encontrado para mover.")
        return 1

    study_records = []
    selected_records = {}
    for accession in accessions:
        studies = client.query_study(
            accession_number=accession,
            additional_attrs=["StudyInstanceUID", "PatientID"],
        )
        if not studies:
            print(f"StudyInstanceUID nao encontrado para {accession}")
            continue
        for study in studies:
            study_uid = study.get("StudyInstanceUID")
            patient_id = study.get("PatientID")
            record = {
                "accession": accession,
                "patient_id": patient_id,
                "study_instance_uid": study_uid,
            }
            study_records.append(record)
            if study_uid and accession not in selected_records:
                selected_records[accession] = record

    patient_accessions: dict[str | None, list[str]] = {}
    for record in study_records:
        patient_id = record["patient_id"]
        accession = record["accession"]
        patient_accessions.setdefault(patient_id, [])
        if accession not in patient_accessions[patient_id]:
            patient_accessions[patient_id].append(accession)

    result_tuples = [(pid, accs) for pid, accs in patient_accessions.items()]
    print("Resultados (patient_id, accession_numbers):")
    print(result_tuples)

    records = list(selected_records.values())
    if not records:
        print("Nenhum StudyInstanceUID encontrado para mover.")
        return 2

    targets = records if args.move_all else records[:1]
    print(
        f"Encontrados {len(records)} accession(s); movendo {len(targets)} estudo(s) para {args.destination_ae}."
    )

    successes = 0
    for record in targets:
        accession = record["accession"]
        study_uid = record["study_instance_uid"]
        if not study_uid:
            print(f"StudyInstanceUID ausente para {accession}")
            continue

        move_result = client.move_study(
            destination_ae=args.destination_ae,
            study_instance_uid=study_uid,
        )
        print(f"Move {accession}: {move_result}")
        logger.info(
            "Move study result",
            extra={
                "extra_data": {
                    "accession": accession,
                    "success": move_result.get("success"),
                    "completed": move_result.get("completed"),
                    "failed": move_result.get("failed"),
                    "warning": move_result.get("warning"),
                }
            },
        )
        if move_result.get("success"):
            successes += 1

    if destination_client:
        verification_failed = False
        print("Verificacao pos-move (C-FIND no destino):")
        for record in targets:
            accession = record["accession"]
            study_uid = record["study_instance_uid"]
            try:
                if study_uid:
                    studies = destination_client.query_study(
                        study_instance_uid=study_uid
                    )
                else:
                    studies = destination_client.query_study(
                        accession_number=accession
                    )
            except Exception as exc:
                studies = []
                logger.warning(
                    "Destination verification failed",
                    extra={"extra_data": {"accession": accession, "error": str(exc)}},
                )
            found = bool(studies)
            if not found:
                verification_failed = True
            print(f"  {accession}: {'ENCONTRADO' if found else 'NAO ENCONTRADO'}")
        if verification_failed:
            print(
                "AVISO: Estudos nao encontrados no destino via C-FIND. "
                "Verifique se o RADIANT aponta o AE de destino para o host/porta corretos."
            )
    else:
        logger.info(
            "Destination verification skipped",
            extra={"extra_data": {"reason": "destination_host/port not provided"}},
        )

    if successes == 0:
        print("Nenhum estudo foi movido com sucesso.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
