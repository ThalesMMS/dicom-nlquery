import json
from typing import List, Dict, Any
from dicom_mcp.dicom_client import DicomClient

# 1. Definição dos Schemas (O que a LLM vê)
DICOM_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_studies",
            "description": "Busca estudos DICOM. TODOS os parâmetros são OPCIONAIS. Omita parâmetros que não precisa filtrar. Chamada sem parâmetros retorna todos os estudos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "ID do paciente (opcional)"},
                    "study_date": {"type": "string", "description": "Data YYYYMMDD ou intervalo YYYYMMDD-YYYYMMDD (opcional)"},
                    "modality": {"type": "string", "description": "MR, CT, US, etc. (opcional)"},
                    "patient_sex": {"type": "string", "description": "F=feminino, M=masculino. Omita se não relevante."},
                    "study_description": {"type": "string", "description": "Filtro de texto com wildcards * (opcional)"}
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
            # Filtra argumentos vazios para não mandar wildcards desnecessários
            sex = args.get("patient_sex")
            if sex and sex not in ("F", "M"):
                sex = None  # Ignora valores inválidos como 'O'
            # Normaliza modalidade (MRI -> MR, etc.)
            modality = args.get("modality", "").upper()
            if modality == "MRI":
                modality = "MR"
            query_args = {
                k: v for k, v in {
                    "patient_id": args.get("patient_id"),
                    "study_date": args.get("study_date"),
                    "modality": modality if modality else None,
                    "patient_sex": sex,
                    "study_description": args.get("study_description"),
                }.items() if v  # Remove strings vazias e None
            }
            results = client.query_study(
                **query_args,
                additional_attrs=["StudyDescription", "PatientBirthDate", "ModalitiesInStudy"]
            )
            # Limita resultados para não estourar contexto da LLM
            summary = results[:20]
            if not summary:
                return json.dumps({"error": "Nenhum estudo encontrado. Tente filtros diferentes."})
            return json.dumps(summary, default=str)

        elif name == "inspect_metadata":
            uid = args.get("study_instance_uid", "").strip()
            if not uid or uid.startswith("<") or not uid[0].isdigit():
                return json.dumps({"error": "study_instance_uid inválido. Use um UID real da busca anterior (ex: 1.2.826.0.1...)"})
            results = client.query_series(
                study_instance_uid=uid,
                additional_attrs=["SeriesDescription", "Modality", "BodyPartExamined"]
            )
            return json.dumps(results, default=str)

        elif name == "move_study":
            uid = args.get("study_instance_uid", "").strip()
            dest = args.get("destination_node", "").strip()
            if not uid or uid.startswith("<") or not uid[0].isdigit():
                return json.dumps({"error": "study_instance_uid inválido. Use um UID real (ex: 1.2.826.0.1...)"})
            if not dest:
                return json.dumps({"error": "destination_node é obrigatório."})
            result = client.move_study(
                destination_ae=dest, 
                study_instance_uid=uid
            )
            return json.dumps(result, default=str)

        return json.dumps({"error": f"Ferramenta {name} desconhecida."})

    except Exception as e:
        return json.dumps({"error": f"Erro de execução: {str(e)}"})