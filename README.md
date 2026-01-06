# dicom-nlquery

Natural language DICOM search (PT-BR) with an LLM-driven query pipeline.

This repo also contains `dicom-mcp`. `dicom-nlquery` delegates DICOM queries to
the `dicom-mcp` server (stdio transport), so the LLM only builds query params and
`dicom-mcp` executes them.

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

# dicom-mcp (server + tools)
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

Certifique-se de que `mcp.config_path` aponta para um dicom-mcp configurado
com o node Orthanc.

## dicom-mcp server

`dicom-nlquery` spawns a `dicom-mcp` server process on demand via stdio. Ensure
`dicom-mcp` is installed in the same venv and that `mcp.config_path` points to a
valid dicom-mcp YAML.

## Agent protocol (stateful)

The DICOM agent follows a deterministic, stateful pipeline:

1. `query_studies` to fetch candidate studies.
2. If multiple results, the agent requires an explicit `study_instance_uid` selection.
3. `query_series` to inspect metadata for the chosen study.
4. `move_study` only after a valid UID is confirmed.

Tool-call protocol:
- Emit tool calls only via `tool_calls` (no JSON in text).
- One tool call per step.
- `destination_node` must be explicit for `move_study` (never inferred).

Operational guardrails (like a default date range) may be applied to protect PACS
performance, but clinical filters are never inferred unless explicitly stated by
the user.

## Move studies with dicom-mcp (NL query -> C-MOVE)

This script parses a natural language query using dicom-nlquery and then uses
dicom-mcp to C-MOVE the first matched study to a destination node.

```bash
cd dicom-nlquery

python scripts/nlquery_move_study.py \
  "mulheres de 20 a 40 anos com cranio" \
  --mcp-config ../dicom-mcp/configuration.yaml \
  --source-node orthanc --destination-node radiant \
  --date-range 20100101-20991231
```

## Configuration

Use `config.yaml` for real PACS or `config-test.yaml` for Orthanc. Example:

```yaml
llm:
  provider: "ollama"
  base_url: "http://127.0.0.1:11434"
  model: "llama3.2:latest"
  temperature: 0
  timeout: 60

guardrails:
  study_date_range_default_days: 180
  max_studies_scanned_default: 2000

mcp:
  command: "dicom-mcp"
  config_path: "../dicom-mcp/configuration.yaml"
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

`--node` usa o dicom-mcp para trocar o node ativo antes da consulta.

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
from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.nl_parser import parse_nl_to_criteria

config = load_config("config.yaml")
criteria = parse_nl_to_criteria("mulheres de 20 a 40 anos", config.llm)
result = execute_search(criteria, mcp_config=config.mcp)
print(result.accession_numbers)
```

## Contributing

- Keep changes small and add tests for new behavior.
- Run `pytest tests/ -v` before submitting.
- Keep PHI out of logs and fixtures.
