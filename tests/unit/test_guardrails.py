from __future__ import annotations

from datetime import date
import logging

from dicom_nlquery.dicom_search import apply_guardrails
from dicom_nlquery.models import GuardrailsConfig


def test_guardrails_defaults_applied() -> None:
    guardrails = GuardrailsConfig()
    today = date(2025, 1, 15)

    date_range, max_studies = apply_guardrails(guardrails, today=today)

    assert date_range == "20240719-20250115"
    assert max_studies == guardrails.max_studies_scanned_default


def test_guardrails_date_range_calculation() -> None:
    guardrails = GuardrailsConfig(study_date_range_default_days=10)
    today = date(2025, 1, 15)

    date_range, _ = apply_guardrails(guardrails, today=today)

    assert date_range == "20250105-20250115"


def test_guardrails_unlimited_logs_warning(caplog) -> None:
    guardrails = GuardrailsConfig()

    with caplog.at_level(logging.WARNING):
        date_range, max_studies = apply_guardrails(guardrails, unlimited=True)

    assert date_range is None
    assert max_studies is None
    assert any("Varredura ilimitada" in record.message for record in caplog.records)
