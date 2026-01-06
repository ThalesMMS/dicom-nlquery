#!/usr/bin/env python3
"""
Natural language DICOM query via dicom-nlquery + dicom-mcp server.

Example:
  python scripts/nlquery_move_study.py \
    "mulheres de 20 a 40 anos com cranio" \
    --mcp-config ../dicom-mcp/configuration.yaml \
    --source-node orthanc \
    --destination-node radiant \
    --date-range 20100101-20991231

Notes:
- Requires Ollama running locally (default: http://127.0.0.1:11434).
- dicom-mcp must be installed in the same venv.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import warnings
from pathlib import Path

import anyio
import yaml

from dicom_nlquery.config import load_config
from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.logging_config import JsonFormatter
from dicom_nlquery.mcp_client import McpSession, build_stdio_server_params
from dicom_nlquery.models import GuardrailsConfig, LLMConfig, McpServerConfig
from dicom_nlquery.nl_parser import parse_nl_to_criteria


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


def _build_temp_mcp_config(args: argparse.Namespace) -> tuple[McpServerConfig, str, str, str]:
    if not args.destination_host or not args.destination_port:
        raise ValueError("destination-host e destination-port sao obrigatorios sem --mcp-config")

    config_data = {
        "nodes": {
            "source": {
                "host": args.host,
                "port": args.port,
                "ae_title": args.called_aet,
                "description": "NL query source",
            },
            "destination": {
                "host": args.destination_host,
                "port": args.destination_port,
                "ae_title": args.destination_ae,
                "description": "NL query destination",
            },
        },
        "current_node": "source",
        "calling_aet": args.calling_aet,
    }

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    Path(temp_file.name).write_text(yaml.safe_dump(config_data), encoding="utf-8")
    temp_file.close()

    mcp_config = McpServerConfig(
        command=args.mcp_command,
        config_path=temp_file.name,
        cwd=args.mcp_cwd,
    )
    return mcp_config, "source", "destination", temp_file.name


def _resolve_mcp_settings(
    args: argparse.Namespace,
    config,
) -> tuple[McpServerConfig, str | None, str, str | None]:
    temp_path = None
    if args.mcp_config:
        mcp = McpServerConfig(
            command=args.mcp_command,
            config_path=str(Path(args.mcp_config).resolve()),
            cwd=args.mcp_cwd,
        )
        return mcp, args.source_node, _require_destination_node(args), temp_path

    if config and config.mcp:
        mcp = config.mcp
        if args.mcp_command:
            mcp = mcp.model_copy(update={"command": args.mcp_command})
        if args.mcp_cwd:
            mcp = mcp.model_copy(update={"cwd": args.mcp_cwd})
        return mcp, args.source_node, _require_destination_node(args), temp_path

    mcp, source_node, destination_node, temp_path = _build_temp_mcp_config(args)
    return mcp, source_node, destination_node, temp_path


def _require_destination_node(args: argparse.Namespace) -> str:
    if not args.destination_node:
        raise ValueError("--destination-node e obrigatorio quando usa dicom-mcp config existente")
    return args.destination_node


async def _move_studies(
    accessions: list[str],
    server_config: McpServerConfig,
    source_node: str | None,
    destination_node: str,
    move_all: bool,
    logger: logging.Logger,
) -> int:
    server_params = build_stdio_server_params(server_config)
    async with McpSession(server_params) as client:
        if source_node:
            await client.switch_dicom_node(source_node)

        study_records = []
        selected_records = {}
        for accession in accessions:
            studies = await client.query_studies(
                accession_number=accession,
                additional_attributes=["StudyInstanceUID", "PatientID"],
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

        targets = records if move_all else records[:1]
        print(
            f"Encontrados {len(records)} accession(s); movendo {len(targets)} estudo(s) para {destination_node}."
        )

        successes = 0
        for record in targets:
            accession = record["accession"]
            study_uid = record["study_instance_uid"]
            if not study_uid:
                print(f"StudyInstanceUID ausente para {accession}")
                continue

            move_result = await client.move_study(
                destination_node=destination_node,
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

        if destination_node:
            verification_failed = False
            await client.switch_dicom_node(destination_node)
            print("Verificacao pos-move (C-FIND no destino):")
            for record in targets:
                accession = record["accession"]
                study_uid = record["study_instance_uid"]
                try:
                    if study_uid:
                        studies = await client.query_studies(
                            study_instance_uid=study_uid
                        )
                    else:
                        studies = await client.query_studies(
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
                    "Verifique o node de destino na configuracao do dicom-mcp."
                )

        if successes == 0:
            print("Nenhum estudo foi movido com sucesso.")
            return 2

        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NL query via dicom-nlquery + dicom-mcp server",
    )
    parser.add_argument("query", help="Natural language query (PT-BR)")
    parser.add_argument("--config", default=None, help="dicom-nlquery config.yaml")
    parser.add_argument("--host", default="localhost", help="DICOM SCP host (legacy)")
    parser.add_argument("--port", type=int, default=11112, help="DICOM SCP port (legacy)")
    parser.add_argument("--called-aet", default="RADIANT", help="Called AE title (legacy)")
    parser.add_argument("--calling-aet", default="MCPSCU", help="Calling AE title (legacy)")
    parser.add_argument(
        "--destination-ae",
        default="MONAI-DEPLOY",
        help="Destination AE title for C-MOVE (legacy)",
    )
    parser.add_argument(
        "--destination-host",
        default=None,
        help="Destination host for temporary dicom-mcp config",
    )
    parser.add_argument(
        "--destination-port",
        type=int,
        default=None,
        help="Destination port for temporary dicom-mcp config",
    )
    parser.add_argument("--source-node", default=None, help="dicom-mcp source node name")
    parser.add_argument("--destination-node", default=None, help="dicom-mcp destination node name")
    parser.add_argument("--mcp-config", default=None, help="dicom-mcp configuration file")
    parser.add_argument("--mcp-command", default="dicom-mcp", help="dicom-mcp command")
    parser.add_argument("--mcp-cwd", default=None, help="Working directory for dicom-mcp")
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
    else:
        llm_config = LLMConfig(
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            model="llama3.2:latest",
            temperature=0,
            timeout=60,
        )
        guardrails_config = GuardrailsConfig()

    if args.llm_base_url:
        llm_config = llm_config.model_copy(update={"base_url": args.llm_base_url})
    if args.llm_model:
        llm_config = llm_config.model_copy(update={"model": args.llm_model})

    try:
        mcp_config, source_node, destination_node, temp_path = _resolve_mcp_settings(
            args, config
        )
    except Exception as exc:
        print(f"Erro ao resolver dicom-mcp: {exc}")
        return 2

    try:
        criteria = parse_nl_to_criteria(args.query, llm_config)
    except Exception as exc:
        print(f"Erro ao parsear a consulta: {exc}")
        return 3
    logger.info("Criteria parsed", extra={"extra_data": criteria.model_dump()})

    try:
        result = execute_search(
            criteria,
            mcp_config=mcp_config,
            date_range=args.date_range,
            max_studies=args.max_studies,
            unlimited=args.unlimited,
            guardrails_config=guardrails_config,
            logger=logger,
            node_name=source_node,
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

    try:
        exit_code = anyio.run(
            _move_studies,
            accessions,
            mcp_config,
            source_node,
            destination_node,
            args.move_all,
            logger,
        )
    except Exception as exc:
        print(f"Erro ao mover estudos: {exc}")
        return 2
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
