from __future__ import annotations

from dicom_nlquery.confirmation import (
    build_confirmation_message,
    classify_confirmation_response,
)
from dicom_nlquery.models import ConfirmationConfig, ResolvedRequest


def test_confirmation_classification() -> None:
    config = ConfirmationConfig(accept_tokens=["yes", "y"], reject_tokens=["no"])

    assert classify_confirmation_response("YES", config) == "accept"
    assert classify_confirmation_response(" no ", config) == "reject"
    assert classify_confirmation_response("maybe", config) == "invalid"


def test_confirmation_message_renders_tokens() -> None:
    config = ConfirmationConfig(accept_tokens=["yes"], reject_tokens=["no"])
    request = ResolvedRequest(filters={"patient_id": "123"})

    message = build_confirmation_message(request, config)

    assert "yes" in message
    assert "no" in message
    assert "patient_id" in message
