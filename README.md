# dicom-nlquery

Natural language DICOM search (PT-BR) with a deterministic filtering pipeline.

## Overview

`dicom-nlquery` converts a natural language query into structured DICOM criteria
and runs C-FIND against a PACS (or Orthanc) to return matching AccessionNumbers.
It keeps the LLM local (Ollama) and applies guardrails for performance and
privacy.

## Installation

```bash
cd dicom-nlquery

# Create venv (optional but recommended)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

## Quick Start

```bash
# Verify Ollama
curl -s http://127.0.0.1:11434/api/tags

# Dry run (no PACS query)
dicom-nlquery dry-run "mulheres de 20 a 40 anos com cranio"

# Execute against a DICOM node
dicom-nlquery execute --date-range 20190101-20210101 \
  "mulheres de 20 a 40 anos com cranio"
```

## Configuration

Create `config.yaml` (or use `config-test.yaml` for Orthanc):

```yaml
nodes:
  orthanc:
    host: "localhost"
    port: 4242
    ae_title: "ORTHANC"
    description: "Orthanc Docker for tests"

current_node: "orthanc"
calling_aet: "NLQUERY"

llm:
  provider: "ollama"
  base_url: "http://127.0.0.1:11434"
  model: "llama3.2:latest"
  temperature: 0
  timeout: 60

guardrails:
  study_date_range_default_days: 180
  max_studies_scanned_default: 2000

matching:
  head_keywords: ["cranio", "cabeca", "head", "brain"]
  synonyms:
    axial: ["axial", "ax"]
    pos: ["pos", "post"]
```

## Usage

```bash
# Dry run

dicom-nlquery dry-run "mulheres de 30 anos com cranio"

# Execute with JSON output

dicom-nlquery execute --json "exames de cranio"

# Execute with custom date range

dicom-nlquery execute --date-range 20240101-20241231 "exames de 2024"

# Override node

dicom-nlquery execute --node orthanc "todos os exames"
```

### Exit Codes

- 0: Success with results
- 1: Success without results
- 2: DICOM error
- 3: Configuration or LLM error

## Safety & Privacy

- PHI fields are masked in logs: `PatientName`, `PatientID`, `PatientBirthDate`.
- Guardrails apply a default date range and study scan limit.
- `--unlimited` is available but logs a warning.
- Logs avoid raw PHI and only include aggregates or masked data.

## Testing

```bash
# Unit tests
cd dicom-nlquery
pytest tests/unit/ -v

# Integration tests (Orthanc Docker)
docker compose -f tests/docker-compose.yml up -d
pytest tests/integration/ -v -m integration
docker compose -f tests/docker-compose.yml down

# Full suite
pytest tests/ -v
```

## Troubleshooting

- "LLM nao disponivel": Ensure Ollama is running (`ollama serve`) and
  `llama3.2:latest` is installed.
- "Falha na associacao DICOM": Check AE titles, host/port, firewall, and
  Orthanc `DicomModalities` entries.
- "Nenhum resultado": Expand date range with `--date-range` and validate
  criteria with `dry-run`.

## API Reference

Minimal programmatic usage:

```python
from dicom_nlquery.config import load_config
from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.nl_parser import parse_nl_to_criteria
from dicom_nlquery.dicom_client import DicomClient

config = load_config("config.yaml")
criteria = parse_nl_to_criteria("mulheres de 20 a 40 anos", config.llm)
node = config.nodes[config.current_node]
client = DicomClient(node.host, node.port, config.calling_aet, node.ae_title)
result = execute_search(criteria, client, matching_config=config.matching)
print(result.accession_numbers)
```

## Contributing

- Keep changes small and add tests for new behavior.
- Run `pytest tests/ -v` before submitting.
- Keep PHI out of logs and fixtures.
