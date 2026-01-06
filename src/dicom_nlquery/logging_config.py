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


PHI_FIELDS = {"PatientName", "PatientID", "PatientBirthDate"}


def mask_phi(value: Any) -> Any:
    if isinstance(value, dict):
        masked: dict[str, Any] = {}
        for key, item in value.items():
            if key in PHI_FIELDS and item:
                masked[key] = "***"
            else:
                masked[key] = mask_phi(item)
        return masked
    if MultiValue is not None and isinstance(value, MultiValue):
        return [mask_phi(item) for item in value]
    if isinstance(value, (list, tuple, set)):
        return [mask_phi(item) for item in value]
    if PersonName is not None and isinstance(value, PersonName):
        return "***"
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
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


def configure_logging(level: str | int = "INFO") -> logging.Logger:
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
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    return logging.getLogger("dicom_nlquery")
