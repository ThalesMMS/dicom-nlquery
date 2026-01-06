import json
import re
import unicodedata
from typing import List, Dict, Any
from .dicom_client import DicomClient

# Mantenha o DICOM_TOOLS_SCHEMA igual...
DICOM_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_studies",
            "description": "Busca estudos DICOM. Retorna lista de UIDs. Use series_description para filtrar por características de série (ex: contraste, difusão, T1, T2).",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "study_date": {"type": "string", "description": "YYYYMMDD ou intervalo"},
                    "modality": {"type": "string", "description": "MR, CT, US, CR"},
                    "patient_sex": {"type": "string", "enum": ["F", "M"], "description": "USE APENAS SE TIVER CERTEZA."},
                    "study_description": {"type": "string", "description": "Wildcards (*) para descrição do ESTUDO"},
                    "series_description": {"type": "string", "description": "Wildcards (*) para descrição da SÉRIE. Use para filtrar por contraste (*gad*, *contrast*), sequências (*T1*, *DWI*), etc."}
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

MODALITY_MAP = {
    "RM": "MR",
    "MRI": "MR",
    "RESSONANCIA": "MR",
    "RESSONANCIA MAGNETICA": "MR",
    "RESSONANCIA MAGNÉTICA": "MR",
    "TC": "CT",
    "TOMOGRAFIA": "CT",
    "USG": "US",
    "ULTRASSOM": "US",
}

ALLOWED_MODALITIES = {"MR", "CT", "US", "CR", "DX", "SR", "PDF"}
UID_RE = re.compile(r"^\d+(?:\.\d+)+$")


def _normalize_text(value: str) -> str:
    """Remove acentos para normalizar comparações de string."""
    return "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))


def _normalize_modality(modality: str) -> str:
    mod = _normalize_text(str(modality)).strip().upper()
    mod = MODALITY_MAP.get(mod, mod)
    return mod


def _valid_uid(uid: str) -> bool:
    return bool(UID_RE.match(uid))

def execute_tool(name: str, args: Dict, client: DicomClient) -> str:
    try:
        if name == "search_studies":
            # Limpa parâmetros vazios
            params = {k: v for k, v in args.items() if v}

            # Normaliza modalidade (sinônimos comuns -> códigos DICOM)
            if "modality" in params:
                mod = _normalize_modality(params["modality"])
                if mod and mod not in ALLOWED_MODALITIES:
                    return json.dumps({"error": f"Modality inválida: {mod}. Use {sorted(ALLOWED_MODALITIES)}"})
                params["modality"] = mod
            
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

            # Fallback simples para “feto/fetal”: usa wildcard que cobre ambos
            if "study_description" in params:
                desc_norm = _normalize_text(str(params["study_description"])).lower()
                if "feto" in desc_norm and "fet" not in desc_norm:
                    params["study_description"] = "*fet*"

            # Extrai filtro de série (será aplicado após query de estudos)
            series_filter = params.pop("series_description", None)

            try:
                results = client.query_studies(**params)
            except Exception as e:
                return f"ERRO DE CONEXÃO: {str(e)}"

            # Se há filtro de série, filtra estudos que contenham séries correspondentes
            if series_filter and results:
                filtered_results = []
                for study in results:
                    uid = study.get("StudyInstanceUID")
                    if not uid:
                        continue
                    try:
                        series_list = client.query_series(
                            study_instance_uid=uid,
                            series_description=series_filter
                        )
                        if series_list:
                            # Adiciona info das séries encontradas ao estudo
                            study["_matching_series"] = [
                                s.get("SeriesDescription", "") for s in series_list[:5]
                            ]
                            filtered_results.append(study)
                    except Exception:
                        pass
                results = filtered_results
                if not results:
                    filtros_usados = ", ".join(list(params.keys()) + ["series_description"])
                    return f"STATUS: Nenhum resultado encontrado com os filtros [{filtros_usados}]. Tente remover filtros ou usar wildcards diferentes."

            if not results:
                filtros_usados = ", ".join(params.keys())
                return f"STATUS: Nenhum resultado encontrado com os filtros [{filtros_usados}]. Tente remover filtros."

            # Resumo eficiente para LLM
            summary = []
            for study in results[:10]:
                item = {
                    "UID": study.get("StudyInstanceUID"),
                    "Date": study.get("StudyDate"),
                    "Modality": study.get("ModalitiesInStudy"),
                    "Desc": study.get("StudyDescription"),
                    "Patient": study.get("PatientName", "ANON")
                }
                # Inclui séries correspondentes se filtro de série foi usado
                if "_matching_series" in study:
                    item["MatchingSeries"] = study["_matching_series"]
                summary.append(item)
            return json.dumps(summary, indent=2)

        elif name == "inspect_metadata":
            uid = str(args.get("study_instance_uid", "")).strip()
            if not uid or "<" in uid:
                return "ERRO: UID inválido. Faça uma busca primeiro."
            if not _valid_uid(uid):
                return json.dumps({"error": f"study_instance_uid inválido: {uid}. Use formato 1.2.826.0.1..."})
            
            try:
                series = client.query_series(study_instance_uid=uid)
            except Exception as e:
                return f"ERRO: {str(e)}"
                
            if not series:
                return "STATUS: Estudo vazio."
            
            return json.dumps(series[:15], indent=2, default=str)

        elif name == "move_study":
            uid = str(args.get("study_instance_uid", "")).strip()
            dest = args.get("destination_node", "")
            
            if not uid or not dest:
                return "ERRO: Parâmetros faltando."
            if not _valid_uid(uid):
                return json.dumps({"error": f"study_instance_uid inválido: {uid}. Use formato 1.2.826.0.1..."})

            try:
                res = client.move_study(destination_node=dest, study_instance_uid=uid)
                return json.dumps(res, indent=2)
            except Exception as e:
                return f"ERRO C-MOVE: {str(e)}"

    except Exception as e:
        return f"ERRO SISTÊMICO: {str(e)}"
    
    return f"Ferramenta {name} desconhecida."
