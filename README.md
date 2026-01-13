# dicom-nlquery

Natural language DICOM search (EN) with an LLM-driven query pipeline.

This repo also contains `dicom-mcp`. `dicom-nlquery` delegates DICOM queries to
the `dicom-mcp` server (stdio transport), so the LLM only builds query params and
`dicom-mcp` executes them.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- OpenAI-compatible LLM server (vLLM, OpenAI, Ollama, or LM Studio)
- Docker (only for integration tests / Orthanc demo)

## Quick checklist

1) Create the shared venv and install packages:

```bash
cd dicom-nlquery
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install -e ../dicom-mcp
```

2) Make sure your LLM endpoint is running (vLLM/Ollama/LM Studio).

3) Run the CLI:

```bash
uv run dicom-nlquery dry-run "cranial exams"
uv run dicom-nlquery execute --date-range 20240101-20241231 "exams from 2024"
```

`dicom-nlquery` starts `dicom-mcp` on demand via stdio, so you usually do not
need to run the server in a separate terminal.

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
# Start Orthanc
cd dicom-nlquery
docker compose -f tests/docker-compose.yml up -d

# Dry-run (no PACS query)
uv run dicom-nlquery dry-run "women ages 20 to 40 with cranial MR"

# Execute (C-FIND)
uv run dicom-nlquery execute --date-range 20190101-20210101 \
  "women ages 20 to 40 with cranial MR"
```

`execute` resolves nodes via `list_dicom_nodes` and requires explicit confirmation
before any MCP tool call (TTY required). If a destination node is present in the
query, it runs `query_studies -> query_series -> move_study` after confirmation.

Make sure `mcp.config_path` points to a dicom-mcp configuration that includes
the Orthanc node.

## dicom-mcp server

`dicom-nlquery` spawns a `dicom-mcp` server process on demand via stdio. Ensure
`dicom-mcp` is installed in the same venv and that `mcp.config_path` points to a
valid dicom-mcp YAML.

## Agent protocol (stateful)

Note: `dicom-nlquery/src/dicom_nlquery/agent.py` and
`dicom-nlquery/scripts/run_agent.py` are marked for potential removal if the
workflow no longer needs the agent loop.

The DICOM agent follows a deterministic, stateful pipeline:

1. Resolve nodes via registry and request confirmation.
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

`dicom-nlquery execute` now supports move flows (when a destination node is
present in the query). For safety, the CLI aborts if multiple studies match and
asks you to refine the search.

This script parses a natural language query using dicom-nlquery and then uses
dicom-mcp to C-MOVE the first matched study to a destination node.

```bash
cd dicom-nlquery

uv run python scripts/nlquery_move_study.py \
  "women ages 20 to 40 with cranial MR" \
  --mcp-config ../configs/dicom.yaml \
  --source-node orthanc --destination-node radiant \
  --date-range 20100101-20991231
```

## Configuration

Use `config.yaml` for real PACS or `config-test.yaml` for Orthanc.

### LLM Configuration

The system supports multiple LLM backends via a unified client interface:

| Provider | API Format | Config Value |
|----------|------------|--------------|
| vLLM | OpenAI-compatible | `openai` |
| OpenAI | OpenAI native | `openai` |
| Ollama | Ollama native | `ollama` |
| LM Studio | Ollama-compatible | `lmstudio` |

LLM settings live in `configs/llm.yaml` (production) or `configs/llm-test.yaml`
(local/test). Reference the file from `dicom-nlquery/config.yaml` via `llm_path`.

**vLLM (Production, configs/llm.yaml):**

```yaml
provider: "openai"
base_url: "http://100.100.101.1:8001"
model: "default"
temperature: 0.1
timeout: 60
max_tokens: 1024
max_completion_tokens: 1024
response_format: "json_schema"
stop:
  - "<|eot_id|>"
# api_key: "your-api-key"  # Optional for vLLM
```

Adjust `stop` tokens to match the deployed model if it uses a different end-of-turn marker.
If the backend rejects `json_schema`, set `response_format` to `json_object`.

**Ollama (Local Development, configs/llm-test.yaml):**

```yaml
provider: "ollama"
base_url: "http://127.0.0.1:11434"
model: "llama3.2:latest"
temperature: 0.1
timeout: 60
```

### Full Configuration Example

```yaml
llm_path: "../configs/llm.yaml"

mcp:
  command: "dicom-mcp"
  config_path: "../configs/dicom.yaml"
  tool_timeout_seconds: 30
  retry:
    max_attempts: 3
    backoff_seconds: [0.5, 1.0, 2.0]

guardrails:
  study_date_range_default_days: 180
  max_studies_scanned_default: 2000
  search_timeout_seconds: 120

search_pipeline:
  enabled: true
  structured_first: true
  max_attempts: 12
  max_rewrites: 10
  series_probe_enabled: false
  series_probe_limit: 50
  wildcard_modes: ["contains", "token_chain", "startswith"]

lexicon:
  path: "configs/lexicon.en-US.yaml"

rag:
  enable: false
  index_path: "../pacs-rag/data/pacs_terms.sqlite"
  top_k: 10
  min_score: 0.2
  provider: "hash"
  embed_dim: 64

ranking:
  enabled: true
  text_match_weight: 0.7
  recency_weight: 0.3

resolver:
  enabled: true
  require_confirmation: true
  confirmation:
    accept_tokens: ["yes", "y"]
    reject_tokens: ["no", "n"]
    prompt_template: |
      Mode: {mode}
      Source: {source_node}
      Destination: {destination_node}
      Filters:
      {filters}
      Confirm? ({accept_tokens}/{reject_tokens})
    invalid_response: "Invalid response. Use: {accept_tokens} or {reject_tokens}."
    correction_prompt: "Enter the corrected query:"
    cancel_message: "Operation cancelled."
    max_invalid_responses: 2
    max_rejections: 2
```

## Search Pipeline Guide

The pipeline is deterministic and bounded to avoid query explosions while still
handling lexical variation (e.g., "fetus" vs "fetal"). Stages run in order until
results are found or guardrails stop further attempts.

1) **Structured-first**
- Sends `query_studies` with structured filters (sex/date/modality/IDs) but no
  description constraint.
- Applies **lexicon-aware local matching** on `StudyDescription` and optionally
  probes series (when `series_probe_enabled=true`).

2) **Wildcards**
- Tries deterministic wildcard patterns for the original description:
  `*term*`, `*t1*t2*`, `term*`, and optional headword.

3) **Lexicon rewrites (beam-limited)**
- Expands tokens using the lexicon and generates bounded candidates via a
  beam-style search (`max_rewrites`).
- Keeps ordering deterministic and avoids combinatorial blow-ups.

4) **RAG rewrites (optional)**
- If enabled, `pacs-rag` suggests real PACS terms; each suggestion is tried as a
  wildcard description.
- Suggestions are cached (LRU) to reduce repeated embedding calls.

## Guardrails & Ranking Decisions

- **Guardrails**: default date range + max studies scanned + total time budget.
  `--unlimited` removes the scan cap but logs a warning.
- **Ranking**: results are ordered by text match overlap and recency (weights
  configurable via `ranking`).
- **No inferred clinical filters**: only explicit fields are used; lexicon/RAG
  only influence text matching and rewrites.

## Observability

`SearchStats` includes:
- `attempts_run`, `successful_stage`, `rewrites_tried`, `stages_tried`
- `studies_scanned`, `studies_matched`, `limit_reached`, `execution_time_seconds`

CLI prints stage/rewrites when no results are found; JSON output always includes
full stats.

## Usage

All commands should be run with `uv run` to ensure the correct environment.

```bash
cd dicom-nlquery

# Dry-run (parse only, no PACS query)
uv run dicom-nlquery dry-run "women age 30 with cranial MR"

# Dry-run output includes node_matches when MCP is configured

# Execute with JSON output
uv run dicom-nlquery execute --json "cranial exams"

# Custom date range
uv run dicom-nlquery execute --date-range 20240101-20241231 "exams from 2024"

# Override node
uv run dicom-nlquery execute --node orthanc "all exams"

# Verbose mode with LLM debug output
uv run dicom-nlquery -v --llm-debug execute "chest CT"
```

`--node` uses dicom-mcp to switch the active node before the query.

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

All tests should be run with `uv run`.

```bash
cd dicom-nlquery

# Unit tests
uv run pytest tests/unit/ -v

# Integration tests (requires Orthanc)
docker compose -f tests/docker-compose.yml up -d
uv run pytest tests/integration/ -v -m integration
docker compose -f tests/docker-compose.yml down

# Full suite
uv run pytest tests/ -v

# Run specific test files
uv run pytest tests/unit/test_node_registry.py -v
uv run pytest tests/unit/test_resolver.py -v
uv run pytest tests/unit/test_confirmation.py -v
```

## PACS Lexicon & RAG

`pacs-rag` can ingest real PACS terminology into a local SQLite index and export
a starter lexicon file for manual curation.

```bash
cd pacs-rag

# Build or update the index (requires a configured MCP server)
uv run pacs-rag ingest-mcp \
  --mcp-command dicom-mcp \
  --config-path ../configs/dicom.yaml \
  --index data/pacs_terms.sqlite \
  --study-date 20240101-20241231

# Export a lexicon skeleton (review and add synonyms)
uv run pacs-rag export-lexicon \
  --index data/pacs_terms.sqlite \
  --output ../dicom-nlquery/configs/lexicon.generated.yaml \
  --min-count 2
```

Then point `lexicon.path` to the generated file and keep RAG optional:

```yaml
lexicon:
  path: "configs/lexicon.generated.yaml"
rag:
  enable: true
```

The generated YAML includes:
- `synonyms`: empty buckets for manual synonym curation
- `ngrams`: frequent bi-grams across PACS descriptions
- `clusters`: simple term clusters based on token overlap (review and edit)

## Troubleshooting

- "LLM not available": ensure your LLM server is running and accessible.
  - For vLLM: check that the server is running at the configured `base_url`
  - For Ollama: ensure `ollama serve` is running and the model is installed
- "DICOM association failed": check AE titles, host/port, firewall, and
  Orthanc `DicomModalities` entries.
- "No results": expand the date range and validate criteria with
  `dry-run`.

## API Reference

```python
from dicom_nlquery.config import load_config
from dicom_nlquery.dicom_search import execute_search
from dicom_nlquery.nl_parser import parse_nl_to_criteria

config = load_config("config.yaml")
criteria = parse_nl_to_criteria("women ages 20 to 40", config.llm)
result = execute_search(criteria, mcp_config=config.mcp)
print(result.accession_numbers)
```

## Contributing

- Keep changes small and add tests for new behavior.
- Run `uv run pytest tests/ -v` before submitting.
- Keep PHI out of logs and fixtures.
