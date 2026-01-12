import pytest


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def chat(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
        return self._response


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient(
        '{\"study\": {'
        '\"patient_id\": null, '
        '\"patient_sex\": \"F\", '
        '\"patient_birth_date\": null, '
        '\"study_date\": null, '
        '\"modality_in_study\": null, '
        '\"study_description\": \"cranial\", '
        '\"accession_number\": null, '
        '\"study_instance_uid\": null'
        '}, \"series\": null}'
    )


@pytest.fixture
def fake_llm_invalid() -> FakeLLMClient:
    return FakeLLMClient(
        '{\"study\": {\"patient_sex\": \"X\", \"study_description\": \"cranial\"}}'
    )


@pytest.fixture
def fake_llm_extra() -> FakeLLMClient:
    return FakeLLMClient(
        '{\"study\": {\"patient_sex\": \"F\"}, \"unexpected\": 123}'
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests that require external services",
    )
    config.addinivalue_line("markers", "slow: tests that are slow to run")
