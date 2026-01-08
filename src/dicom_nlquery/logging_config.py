from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

try:
    from pydicom.multival import MultiValue
    from pydicom.valuerep import PersonName
except Exception:  # pragma: no cover - optional typing support
    MultiValue = None
    PersonName = None


def _normalize_key(raw: object) -> str:
    return "".join(char for char in str(raw).lower() if char.isalnum())


PHI_FIELDS = {
    "AccessionNumber",
    "MedicalRecordNumber",
    "OtherPatientIDs",
    "OtherPatientNames",
    "PatientAddress",
    "PatientBirthDate",
    "PatientBirthName",
    "PatientBirthTime",
    "PatientComments",
    "PatientID",
    "PatientMotherBirthName",
    "PatientName",
    "PatientSSN",
    "PatientSocialSecurityNumber",
    "PatientTelephoneNumbers",
}

_PHI_KEY_NORMALIZED = {
    *{_normalize_key(key) for key in PHI_FIELDS},
    "accession",
    "mrn",
    "medicalrecordnumber",
    "patientdateofbirth",
    "patientdob",
    "patientemail",
    "patientmrn",
    "patientphonenumber",
    "patientphone",
    "patienttelephonenumber",
}

_PATIENT_CONTAINER_KEYS = {"patient", "patients", "subject", "subjects"}
_PATIENT_CHILD_KEYS = {
    "address",
    "birthdate",
    "birthname",
    "birthtime",
    "comment",
    "comments",
    "dateofbirth",
    "dob",
    "email",
    "id",
    "identifier",
    "medicalrecordnumber",
    "mrn",
    "motherbirthname",
    "name",
    "otherids",
    "othernames",
    "phone",
    "phonenumber",
    "socialsecuritynumber",
    "ssn",
    "telephone",
    "telephonenumber",
    "telephonenumbers",
}


def _should_mask_key(key: str, path: tuple[str, ...]) -> bool:
    if not key:
        return False
    if key in _PHI_KEY_NORMALIZED:
        return True
    if any(part in _PATIENT_CONTAINER_KEYS for part in path):
        return key in _PATIENT_CHILD_KEYS
    return False


def mask_phi(value: Any, path: tuple[str, ...] | None = None) -> Any:
    if path is None:
        path = ()
    if isinstance(value, dict):
        masked: dict[str, Any] = {}
        normalized_path = tuple(_normalize_key(part) for part in path)
        for key, item in value.items():
            key_norm = _normalize_key(key)
            if _should_mask_key(key_norm, normalized_path) and item:
                masked[key] = "***"
            else:
                masked[key] = mask_phi(item, path + (str(key),))
        return masked
    if MultiValue is not None and isinstance(value, MultiValue):
        return [mask_phi(item, path) for item in value]
    if isinstance(value, (list, tuple, set)):
        return [mask_phi(item, path) for item in value]
    if PersonName is not None and isinstance(value, PersonName):
        return "***"
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        try:
            return mask_phi(value.model_dump(), path)
        except Exception:
            return str(value)
    return str(value)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_data = getattr(record, "extra_data", None)
        if extra_data is not None:
            payload["extra"] = mask_phi(extra_data)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


class _ConsoleFormatter(logging.Formatter):
    def __init__(self, show_extra: bool) -> None:
        super().__init__("%(levelname)s: %(message)s")
        self._show_extra = show_extra

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not self._show_extra:
            return message
        extra_data = getattr(record, "extra_data", None)
        if extra_data is None:
            return message
        try:
            extra_json = json.dumps(mask_phi(extra_data), ensure_ascii=True)
        except TypeError:
            extra_json = str(mask_phi(extra_data))
        return f"{message} | {extra_json}"


def configure_logging(level: str | int = "INFO", show_extra: bool = False) -> logging.Logger:
    import sys

    class _DynamicStreamHandler(logging.StreamHandler):
        def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
            self.stream = sys.stderr
            super().emit(record)
    
    # Force root logger to DEBUG to capture everything for the file
    # The 'level' argument acts as a minimum for the console if needed, 
    # but we'll stick to fixed levels for consistency:
    # - File: DEBUG (All logs)
    # - Console: INFO (Clean logs)
    
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Clear existing handlers
    if root.handlers:
        root.handlers.clear()

    # 1. File Handler - JSON formatted, verbose (DEBUG)
    file_handler = logging.FileHandler("dicom-nlquery.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    # 2. Console Handler - Text formatted, clean (INFO)
    console_handler = _DynamicStreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_ConsoleFormatter(show_extra))
    root.addHandler(console_handler)

    return logging.getLogger("dicom_nlquery")
