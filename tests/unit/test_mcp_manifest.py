import pytest

from dicom_nlquery.mcp_client import (
    REQUIRED_MANIFEST_VERSION,
    REQUIRED_SCHEMA_VERSION,
    REQUIRED_TOOL_VERSIONS,
    validate_manifest_payload,
)


def _build_manifest(
    *,
    manifest_version: str | None = None,
    schema_version: str | None = None,
    tool_versions: dict[str, str] | None = None,
) -> dict:
    return {
        "manifest_version": manifest_version or REQUIRED_MANIFEST_VERSION,
        "schema_version": schema_version or REQUIRED_SCHEMA_VERSION,
        "server": {"name": "dicom-mcp", "version": "0.1.1"},
        "tools": {
            "required": tool_versions or dict(REQUIRED_TOOL_VERSIONS),
            "optional": {},
        },
    }


def test_validate_manifest_accepts_compatible_manifest() -> None:
    manifest = _build_manifest()
    validate_manifest_payload(manifest)


def test_validate_manifest_rejects_missing_tool() -> None:
    tool_versions = dict(REQUIRED_TOOL_VERSIONS)
    tool_versions.pop("query_series")
    manifest = _build_manifest(tool_versions=tool_versions)
    with pytest.raises(ValueError, match="missing required tool 'query_series'"):
        validate_manifest_payload(manifest)


def test_validate_manifest_rejects_incompatible_schema_version() -> None:
    manifest = _build_manifest(schema_version="2.0")
    with pytest.raises(ValueError, match="schema_version"):
        validate_manifest_payload(manifest)
