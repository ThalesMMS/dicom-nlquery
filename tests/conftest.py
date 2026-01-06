import pytest


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient(
        '{\"patient\": {\"sex\": \"F\", \"age_min\": 20, \"age_max\": 40}, '
        '\"head_keywords\": [\"cranio\"], '
        '\"required_series\": [], '
        '\"study_narrowing\": {\"modality_in_study\": null, \"study_description_keywords\": null}}'
    )


@pytest.fixture
def fake_llm_invalid() -> FakeLLMClient:
    return FakeLLMClient('{\"patient\": {\"sex\": \"X\"}}')


@pytest.fixture
def fake_llm_extra() -> FakeLLMClient:
    return FakeLLMClient(
        '{\"patient\": {\"sex\": \"F\"}, \"unexpected\": 123, \"head_keywords\": []}'
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests that require external services",
    )
    config.addinivalue_line("markers", "slow: tests that are slow to run")
