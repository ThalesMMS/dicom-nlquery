# dicom-nlquery

Natural language DICOM search (PT-BR) with a deterministic filtering pipeline.

This repo also contains `dicom-mcp`. You can run them side-by-side against the
same Orthanc/PACS. `dicom-nlquery` does not depend on the `dicom-mcp` server,
but if `dicom-mcp` is installed in the same venv it will reuse its DICOM client.

## Prerequisites

- Python 3.12+
- uv
- Ollama running locally (for parsing)
- Docker (only for integration tests / Orthanc demo)

## One-time setup (single venv for both)

```bash
cd dicom-nlquery
uv venv
source .venv/bin/activate

# dicom-nlquery
uv pip install -e ".[dev]"

# optional: dicom-mcp (shared DICOM client + server tools)
uv pip install -e ../dicom-mcp
```

## Run with Orthanc (local demo)

```bash
# start Orthanc
cd dicom-nlquery
docker compose -f tests/docker-compose.yml up -d

# dry-run (no PACS query)
dicom-nlquery dry-run "mulheres de 20 a 40 anos com cranio"

# execute (C-FIND)
dicom-nlquery execute --date-range 20190101-20210101 \
  "mulheres de 20 a 40 anos com cranio"
```

## Run alongside dicom-mcp (optional)

Open a second terminal in the same venv and start the dicom-mcp server:

```bash
cd dicom-mcp

dicom-mcp tests/test_dicom_servers.yaml --transport stdio
```

Notes:
- `dicom-nlquery` does not call the `dicom-mcp` server; it talks directly to the
  DICOM node. Running both is useful when you want MCP tools and NL queries
  against the same Orthanc/PACS.
- Ensure your PACS/Orthanc allows the calling AET (`calling_aet` in config).
  The included Orthanc test compose already allows `NLQUERY` and `TESTSCU`.

## Configuration

Use `config.yaml` for real PACS or `config-test.yaml` for Orthanc. Example:

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
# dry-run
dicom-nlquery dry-run "mulheres de 30 anos com cranio"

# execute with JSON output
dicom-nlquery execute --json "exames de cranio"

# custom date range
dicom-nlquery execute --date-range 20240101-20241231 "exames de 2024"

# override node
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
# unit tests
cd dicom-nlquery
pytest tests/unit/ -v

# integration tests (Orthanc)
docker compose -f tests/docker-compose.yml up -d
pytest tests/integration/ -v -m integration
docker compose -f tests/docker-compose.yml down

# full suite
pytest tests/ -v
```

## Troubleshooting

- "LLM nao disponivel": ensure Ollama is running (`ollama serve`) and
  `llama3.2:latest` is installed.
- "Falha na associacao DICOM": check AE titles, host/port, firewall, and
  Orthanc `DicomModalities` entries.
- "Nenhum resultado": expand the date range and validate criteria with
  `dry-run`.

## API Reference

```python
from dicom_nlquery.config import load_config
from dicom_nlquery.dicom_client import DicomClient
from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.nl_parser import parse_nl_to_criteria

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
