def test_import_works() -> None:
    """
    Given: Project installed
    When: Import dicom_nlquery
    Then: Import succeeds
    """
    import dicom_nlquery

    assert dicom_nlquery is not None
