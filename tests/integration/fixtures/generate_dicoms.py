from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydicom import dcmwrite
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, MRImageStorage, generate_uid


@dataclass(frozen=True)
class SyntheticStudy:
    patient_sex: str
    patient_birth_date: str
    study_date: str
    accession_number: str
    series_description: str
    modality: str
    study_description: str


SYNTHETIC_STUDIES: list[SyntheticStudy] = [
    SyntheticStudy(
        patient_sex="F",
        patient_birth_date="19900101",
        study_date="20200101",
        accession_number="ACC001",
        series_description="AX T1 CRANIO",
        modality="MR",
        study_description="MR cranio study",
    ),
    SyntheticStudy(
        patient_sex="F",
        patient_birth_date="19950601",
        study_date="20200601",
        accession_number="ACC002",
        series_description="SAG T2 CRANIO",
        modality="MR",
        study_description="MR cranio study",
    ),
    SyntheticStudy(
        patient_sex="M",
        patient_birth_date="19850101",
        study_date="20200101",
        accession_number="ACC003",
        series_description="AX T1 TORAX",
        modality="CT",
        study_description="CT torax study",
    ),
    SyntheticStudy(
        patient_sex="F",
        patient_birth_date="19700101",
        study_date="20200101",
        accession_number="ACC004",
        series_description="AX CRANIO",
        modality="CT",
        study_description="CT cranio study",
    ),
    SyntheticStudy(
        patient_sex="F",
        patient_birth_date="20000101",
        study_date="20200101",
        accession_number="ACC005",
        series_description="CRANIO MT POS",
        modality="MR",
        study_description="MR cranio study",
    ),
]


def _sop_class_for_modality(modality: str) -> str:
    if modality.upper() == "MR":
        return MRImageStorage
    return CTImageStorage


def _build_dataset(study: SyntheticStudy, index: int) -> Dataset:
    sop_class_uid = _sop_class_for_modality(study.modality)
    sop_instance_uid = generate_uid()
    study_uid = generate_uid()
    series_uid = generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.PatientName = f"TEST^{index:03d}"
    ds.PatientID = f"TEST{index:03d}"
    ds.PatientBirthDate = study.patient_birth_date
    ds.PatientSex = study.patient_sex

    ds.StudyInstanceUID = study_uid
    ds.StudyDate = study.study_date
    ds.StudyTime = "120000"
    ds.StudyID = f"ST{index:03d}"
    ds.StudyDescription = study.study_description
    ds.AccessionNumber = study.accession_number

    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 1
    ds.Modality = study.modality
    ds.SeriesDescription = study.series_description

    ds.SOPInstanceUID = sop_instance_uid
    ds.SOPClassUID = sop_class_uid
    ds.InstanceNumber = 1

    ds.Rows = 16
    ds.Columns = 16
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = bytes([index % 256] * (16 * 16))

    return ds


def generate_synthetic_dicoms(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, study in enumerate(SYNTHETIC_STUDIES, start=1):
        ds = _build_dataset(study, index)
        path = output_dir / f"synthetic_{index}.dcm"
        dcmwrite(path, ds, write_like_original=False)
        paths.append(path)
    return paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic DICOM files.")
    parser.add_argument("output_dir", help="Directory to write DICOM files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    paths = generate_synthetic_dicoms(output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
