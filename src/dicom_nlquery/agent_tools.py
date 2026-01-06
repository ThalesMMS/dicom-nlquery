import json
from typing import List, Dict, Any
from .dicom_client import DicomClient

# Mantenha o DICOM_TOOLS_SCHEMA igual...
DICOM_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_studies",
            "description": "Busca estudos DICOM. Retorna lista de UIDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "study_date": {"type": "string", "description": "YYYYMMDD ou intervalo"},
                    "modality": {"type": "string", "description": "MR, CT, US, CR"},
                    "patient_sex": {"type": "string", "enum": ["F", "M"], "description": "USE APENAS SE TIVER CERTEZA."},
                    "study_description": {"type": "string", "description": "Wildcards (*)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_metadata",
            "description": "Lista séries de um UID específico.",
            "parameters": {
                "type": "object",
                "properties": {
                    "study_instance_uid": {"type": "string"}
                },
                "required": ["study_instance_uid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_study",
            "description": "Move estudo para destino.",
            "parameters": {
                "type": "object",
                "properties": {
                    "study_instance_uid": {"type": "string"},
                    "destination_node": {"type": "string"}
                },
                "required": ["study_instance_uid", "destination_node"]
            }
        }
    }
]

def execute_tool(name: str, args: Dict, client: DicomClient) -> str:
    try:
        if name == "search_studies":
            # Limpa parâmetros vazios
            params = {k: v for k, v in args.items() if v}
            
            # Corrige nomes de parâmetros comuns (LLM vs Pynetdicom)
            if "modality" in params:
                params["modality_in_study"] = params.pop("modality")
            
            # Validação estrita de sexo
            if "patient_sex" in params:
                sex = str(params["patient_sex"]).upper()
                if sex not in ["M", "F"]:
                    params.pop("patient_sex")
                else:
                    params["patient_sex"] = sex

            try:
                results = client.query_studies(**params)
            except Exception as e:
                return f"ERRO DE CONEXÃO: {str(e)}"

            if not results:
                filtros_usados = ", ".join(params.keys())
                return f"STATUS: Nenhum resultado encontrado com os filtros [{filtros_usados}]. Tente remover filtros."

            # Resumo eficiente para LLM
            summary = []
            for study in results[:10]:
                summary.append({
                    "UID": study.get("StudyInstanceUID"),
                    "Date": study.get("StudyDate"),
                    "Modality": study.get("ModalitiesInStudy"),
                    "Desc": study.get("StudyDescription"),
                    "Patient": study.get("PatientName", "ANON")
                })
            return json.dumps(summary, indent=2)

        elif name == "inspect_metadata":
            uid = args.get("study_instance_uid", "")
            if not uid or "<" in uid:
                return "ERRO: UID inválido. Faça uma busca primeiro."
            
            try:
                series = client.query_series(study_instance_uid=uid)
            except Exception as e:
                return f"ERRO: {str(e)}"
                
            if not series:
                return "STATUS: Estudo vazio."
            
            return json.dumps(series[:15], indent=2, default=str)

        elif name == "move_study":
            uid = args.get("study_instance_uid", "")
            dest = args.get("destination_node", "")
            
            if not uid or not dest:
                return "ERRO: Parâmetros faltando."

            try:
                res = client.move_study(destination_node=dest, study_instance_uid=uid)
                return json.dumps(res, indent=2)
            except Exception as e:
                return f"ERRO C-MOVE: {str(e)}"

    except Exception as e:
        return f"ERRO SISTÊMICO: {str(e)}"
    
    return f"Ferramenta {name} desconhecida."