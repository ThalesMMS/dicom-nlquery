from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


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
    if isinstance(value, list):
        return [mask_phi(item) for item in value]
    return value


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
    if isinstance(level, str):
        level_value = logging.getLevelName(level.upper())
    else:
        level_value = level

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level_value)

    return logging.getLogger("dicom_nlquery")
