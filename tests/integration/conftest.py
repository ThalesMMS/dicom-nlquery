from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import pytest

from dicom_nlquery.dicom_client import DicomClient

from .fixtures.generate_dicoms import SYNTHETIC_STUDIES, generate_synthetic_dicoms


ORTHANC_HOST = os.environ.get("ORTHANC_HOST", "localhost")
ORTHANC_PORT = int(os.environ.get("ORTHANC_PORT", "4242"))
ORTHANC_WEB_PORT = int(os.environ.get("ORTHANC_WEB_PORT", "8042"))
ORTHANC_AET = os.environ.get("ORTHANC_AET", "ORTHANC")
ORTHANC_USERNAME = os.environ.get("ORTHANC_USERNAME", "")
ORTHANC_PASSWORD = os.environ.get("ORTHANC_PASSWORD", "")
ORTHANC_BASE_URL = f"http://{ORTHANC_HOST}:{ORTHANC_WEB_PORT}"


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return self._response




def _http_client() -> httpx.Client:
    auth = None
    if ORTHANC_USERNAME and ORTHANC_PASSWORD:
        auth = (ORTHANC_USERNAME, ORTHANC_PASSWORD)
    return httpx.Client(base_url=ORTHANC_BASE_URL, auth=auth, timeout=5.0)


def _wait_for_orthanc(max_attempts: int = 10, delay: float = 1.0) -> bool:
    for _ in range(max_attempts):
        try:
            with _http_client() as client:
                response = client.get("/system")
                if response.status_code == 200:
                    return True
        except httpx.RequestError:
            pass
        time.sleep(delay)
    return False


def _reset_orthanc() -> None:
    with _http_client() as client:
        response = client.get("/studies")
        response.raise_for_status()
        for study_id in response.json():
            client.delete(f"/studies/{study_id}")


def _upload_dicoms(paths: list[Path]) -> None:
    with _http_client() as client:
        for path in paths:
            response = client.post(
                "/instances",
                content=path.read_bytes(),
                headers={"Content-Type": "application/dicom"},
            )
            response.raise_for_status()


@pytest.fixture(scope="session")
def orthanc_with_data(tmp_path_factory: pytest.TempPathFactory):
    if not _wait_for_orthanc():
        pytest.skip("Orthanc is not available on the expected ports")

    _reset_orthanc()
    output_dir = tmp_path_factory.mktemp("synthetic_dicoms")
    paths = generate_synthetic_dicoms(output_dir)
    _upload_dicoms(paths)

    dicom_client = DicomClient(
        host=ORTHANC_HOST,
        port=ORTHANC_PORT,
        calling_aet="NLQUERY",
        called_aet=ORTHANC_AET,
    )

    context = {
        "client": dicom_client,
        "studies": SYNTHETIC_STUDIES,
        "date_range": "20190101-20210101",
    }

    yield context

    _reset_orthanc()


@pytest.fixture
def fake_llm_female_cranio() -> FakeLLMClient:
    return FakeLLMClient(
        '{\"study\": {'
        '\"patient_id\": null, '
        '\"patient_sex\": \"F\", '
        '\"patient_birth_date\": null, '
        '\"study_date\": null, '
        '\"modality_in_study\": null, '
        '\"study_description\": \"cranio\", '
        '\"accession_number\": null, '
        '\"study_instance_uid\": null'
        '}, \"series\": null}'
    )


@pytest.fixture
def fake_llm_no_match() -> FakeLLMClient:
    return FakeLLMClient(
        '{\"study\": {'
        '\"patient_id\": null, '
        '\"patient_sex\": \"M\", '
        '\"patient_birth_date\": null, '
        '\"study_date\": null, '
        '\"modality_in_study\": null, '
        '\"study_description\": \"abdomen\", '
        '\"accession_number\": null, '
        '\"study_instance_uid\": null'
        '}, \"series\": null}'
    )


@pytest.fixture
def fake_llm_feto() -> FakeLLMClient:
    return FakeLLMClient(
        '{\"study\": {'
        '\"patient_id\": null, '
        '\"patient_sex\": null, '
        '\"patient_birth_date\": null, '
        '\"study_date\": null, '
        '\"modality_in_study\": \"MR\", '
        '\"study_description\": \"feto\", '
        '\"accession_number\": null, '
        '\"study_instance_uid\": null'
        '}, \"series\": null}'
    )


@pytest.fixture
def fake_llm_fetal() -> FakeLLMClient:
    return FakeLLMClient(
        '{\"study\": {'
        '\"patient_id\": null, '
        '\"patient_sex\": null, '
        '\"patient_birth_date\": null, '
        '\"study_date\": null, '
        '\"modality_in_study\": \"MR\", '
        '\"study_description\": \"fetal\", '
        '\"accession_number\": null, '
        '\"study_instance_uid\": null'
        '}, \"series\": null}'
    )
