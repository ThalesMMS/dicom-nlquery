import json
from typing import List, Dict, Any
from dicom_mcp.dicom_client import DicomClient

# 1. Definição dos Schemas (O que a LLM vê)
DICOM_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_studies",
            "description": "Busca inicial ampla por estudos. Use para reduzir o universo de busca (ex: data, modalidade).",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "study_date": {"type": "string", "description": "YYYYMMDD ou intervalo YYYYMMDD-YYYYMMDD"},
                    "modality": {"type": "string", "description": "Modalidade principal (MR, CT, US)"},
                    "patient_sex": {"type": "string", "enum": ["F", "M", "O"]},
                    "study_description": {"type": "string", "description": "Filtro de texto (wildcards * permitidos)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_metadata",
            "description": "Baixa os metadados detalhados (séries e descrições) de um estudo. ESSENCIAL para verificar protocolos clínicos (ex: se tem FLAIR, Contraste, T2).",
            "parameters": {
                "type": "object",
                "properties": {
                    "study_instance_uid": {"type": "string", "description": "UID do estudo candidato"}
                },
                "required": ["study_instance_uid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_study",
            "description": "Move um estudo validado para o destino final.",
            "parameters": {
                "type": "object",
                "properties": {
                    "study_instance_uid": {"type": "string"},
                    "destination_node": {"type": "string", "description": "Nome do node (ex: RADIANT)"}
                },
                "required": ["study_instance_uid", "destination_node"]
            }
        }
    }
]

# 2. Executor (O que o Python faz)
def execute_tool(name: str, args: Dict, client: DicomClient) -> str:
    """Executa a ação cega no DICOM Client e retorna JSON cru."""
    try:
        if name == "search_studies":
            results = client.query_study(
                patient_id=args.get("patient_id"),
                study_date=args.get("study_date"),
                modality=args.get("modality"),
                patient_sex=args.get("patient_sex"),
                study_description=args.get("study_description"),
                # Trazemos campos extras para a LLM decidir se vale a pena inspecionar
                additional_attrs=["StudyDescription", "PatientBirthDate", "ModalitiesInStudy"]
            )
            # Limita resultados para não estourar contexto da LLM
            summary = results[:20] 
            return json.dumps(summary, default=str)

        elif name == "inspect_metadata":
            # Aqui a LLM vai ler "FLAIR", "T2", "DWI" no retorno
            results = client.query_series(
                study_instance_uid=args["study_instance_uid"],
                additional_attrs=["SeriesDescription", "Modality", "BodyPartExamined"]
            )
            return json.dumps(results, default=str)

        elif name == "move_study":
            # A LLM deve passar o AE Title correto, ou resolvemos aqui via config se necessário
            result = client.move_study(
                destination_ae=args["destination_node"], 
                study_instance_uid=args["study_instance_uid"]
            )
            return json.dumps(result, default=str)

        return f"Erro: Ferramenta {name} desconhecida."

    except Exception as e:
        return f"Erro de execução: {str(e)}"