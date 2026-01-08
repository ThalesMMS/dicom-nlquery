from __future__ import annotations

import json
import logging

from dicom_nlquery.logging_config import JsonFormatter, _ConsoleFormatter, mask_phi


def _make_record(extra_data: dict[str, object]) -> logging.LogRecord:
    record = logging.LogRecord(
        name="dicom_nlquery.tests",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Test log",
        args=(),
        exc_info=None,
    )
    record.extra_data = extra_data
    return record


def test_mask_phi_normalizes_keys_and_preserves_non_phi() -> None:
    data = {
        "PatientName": "Jane Doe",
        "patient_name": "Jane Doe",
        "patient-name": "Jane Doe",
        "patient id": "123",
        "PatientBirthDate": "19700101",
        "AccessionNumber": "ACC-42",
        "StudyDescription": "CT HEAD",
        "StudyInstanceUID": "1.2.3.4",
    }

    masked = mask_phi(data)

    assert masked["PatientName"] == "***"
    assert masked["patient_name"] == "***"
    assert masked["patient-name"] == "***"
    assert masked["patient id"] == "***"
    assert masked["PatientBirthDate"] == "***"
    assert masked["AccessionNumber"] == "***"
    assert masked["StudyDescription"] == "CT HEAD"
    assert masked["StudyInstanceUID"] == "1.2.3.4"


def test_mask_phi_nested_patient_context() -> None:
    data = {
        "patient": {"name": "Ana", "id": "123", "dob": "19800101", "sex": "F"},
        "study": {"name": "Brain MRI"},
        "entries": [{"patient": {"email": "ana@example.com"}}, {"name": "Report"}],
    }

    masked = mask_phi(data)

    assert masked["patient"]["name"] == "***"
    assert masked["patient"]["id"] == "***"
    assert masked["patient"]["dob"] == "***"
    assert masked["patient"]["sex"] == "F"
    assert masked["study"]["name"] == "Brain MRI"
    assert masked["entries"][0]["patient"]["email"] == "***"
    assert masked["entries"][1]["name"] == "Report"


def test_console_formatter_masks_extra_data() -> None:
    formatter = _ConsoleFormatter(show_extra=True)
    record = _make_record({"patient_name": "Jane Doe", "study_description": "Head CT"})

    output = formatter.format(record)

    assert "Jane Doe" not in output
    assert "Head CT" in output
    assert "***" in output


def test_json_formatter_masks_extra_data() -> None:
    formatter = JsonFormatter()
    record = _make_record({"patient_name": "Jane Doe", "study_description": "Head CT"})

    payload = json.loads(formatter.format(record))

    assert payload["extra"]["patient_name"] == "***"
    assert payload["extra"]["study_description"] == "Head CT"
