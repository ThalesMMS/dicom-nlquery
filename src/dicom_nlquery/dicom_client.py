from __future__ import annotations

from typing import Iterable

import logging

from pydicom.dataset import Dataset
from pynetdicom import AE
from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind

from .logging_config import mask_phi


class DicomClient:
    def __init__(self, host: str, port: int, calling_aet: str, called_aet: str) -> None:
        self.host = host
        self.port = port
        self.called_aet = called_aet
        self.ae = AE(ae_title=calling_aet)
        self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)
        log = logging.getLogger(__name__)
        log.debug(
            "DicomClient initialized",
            extra={
                "extra_data": {
                    "host": host,
                    "port": port,
                    "called_aet": called_aet,
                    "calling_aet": calling_aet,
                }
            },
        )

    def _find(self, query: Dataset) -> list[dict]:
        log = logging.getLogger(__name__)
        log.debug(
            "C-FIND request",
            extra={"extra_data": {"query": mask_phi(self._dataset_to_dict(query))}},
        )
        assoc = self.ae.associate(self.host, self.port, ae_title=self.called_aet)
        if not assoc.is_established:
            log.error(
                "C-FIND association failed",
                extra={
                    "extra_data": {
                        "host": self.host,
                        "port": self.port,
                        "called_aet": self.called_aet,
                    }
                },
            )
            raise RuntimeError("Failed to establish association with DICOM SCP")

        results: list[dict] = []
        try:
            responses = assoc.send_c_find(query, StudyRootQueryRetrieveInformationModelFind)
            for status, dataset in responses:
                if status and status.Status in {0xFF00, 0xFF01} and dataset:
                    results.append(self._dataset_to_dict(dataset))
        finally:
            assoc.release()
        return results

    def query_study(self, **kwargs) -> list[dict]:
        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"

        if kwargs.get("patient_id"):
            ds.PatientID = kwargs["patient_id"]
        if kwargs.get("patient_sex"):
            ds.PatientSex = kwargs["patient_sex"]
        if kwargs.get("patient_birth_date"):
            ds.PatientBirthDate = kwargs["patient_birth_date"]
        if kwargs.get("study_date"):
            ds.StudyDate = kwargs["study_date"]
        if kwargs.get("modality"):
            ds.ModalitiesInStudy = kwargs["modality"]
        if kwargs.get("study_description"):
            ds.StudyDescription = kwargs["study_description"]
        if kwargs.get("accession_number"):
            ds.AccessionNumber = kwargs["accession_number"]
        if kwargs.get("study_instance_uid"):
            ds.StudyInstanceUID = kwargs["study_instance_uid"]

        attrs = [
            "StudyInstanceUID",
            "StudyDate",
            "StudyDescription",
            "AccessionNumber",
            "ModalitiesInStudy",
        ]
        extra = list(kwargs.get("additional_attrs") or [])
        self._apply_return_keys(ds, attrs + extra)
        return self._find(ds)

    def query_studies(self, **kwargs) -> list[dict]:
        if "modality_in_study" in kwargs and "modality" not in kwargs:
            kwargs = {**kwargs, "modality": kwargs["modality_in_study"]}
            kwargs.pop("modality_in_study", None)
        return self.query_study(**kwargs)

    def query_series(self, study_instance_uid: str, **kwargs) -> list[dict]:
        ds = Dataset()
        ds.QueryRetrieveLevel = "SERIES"
        ds.StudyInstanceUID = study_instance_uid

        if kwargs.get("series_instance_uid"):
            ds.SeriesInstanceUID = kwargs["series_instance_uid"]
        if kwargs.get("modality"):
            ds.Modality = kwargs["modality"]
        if kwargs.get("series_description"):
            ds.SeriesDescription = kwargs["series_description"]

        attrs = [
            "SeriesInstanceUID",
            "StudyInstanceUID",
            "Modality",
            "SeriesNumber",
            "SeriesDescription",
        ]
        extra = list(kwargs.get("additional_attrs") or [])
        self._apply_return_keys(ds, attrs + extra)
        return self._find(ds)

    @staticmethod
    def _apply_return_keys(dataset: Dataset, keys: Iterable[str]) -> None:
        for key in keys:
            if not hasattr(dataset, key):
                setattr(dataset, key, "")

    @staticmethod
    def _dataset_to_dict(dataset: Dataset) -> dict:
        result: dict = {}
        for elem in dataset:
            if elem.VR == "SQ":
                result[elem.keyword] = [
                    DicomClient._dataset_to_dict(item) for item in elem.value
                ]
            else:
                if elem.keyword:
                    try:
                        if elem.VM > 1:
                            result[elem.keyword] = list(elem.value)
                        else:
                            result[elem.keyword] = elem.value
                    except Exception:
                        result[elem.keyword] = str(elem.value)
        return result
