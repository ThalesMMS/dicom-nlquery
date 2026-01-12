# Scripts

Utility scripts for development, testing, and debugging.

## integration_vllm_smoke.py

Smoke test script for validating LLM backend connectivity and response quality.

### Usage

```bash
# Using default config (configs/llm.yaml)
python scripts/integration_vllm_smoke.py

# With custom config
python scripts/integration_vllm_smoke.py --config configs/llm-test.yaml

# With custom timeout threshold
python scripts/integration_vllm_smoke.py --max-time 15

# Verbose output (show LLM responses)
python scripts/integration_vllm_smoke.py --verbose

# Test a single custom query
python scripts/integration_vllm_smoke.py -q "CT chest exams for women"
```

### Test Cases

| Test | Query | Validates |
|------|-------|-----------|
| CT angiogram + routing + age | `"studies from year 2000 until 2022 of CT chest angiograms from ORTHANC to RADIANT, patients age 20 to 80"` | study_description, study_date, patient_birth_date, modality |
| MRI with sex filter | `"cranial MRI for women ages 30 to 50"` | modality, patient_birth_date, patient_sex |
| Ultrasound with body part | `"obstetric ultrasound exams"` | modality, study_description |
| Simple modality | `"CT exams"` | modality |
| Patient name | `"exams for patient SILVA"` | patient_name |

### Exit Codes

- `0` - All tests passed
- `1` - One or more tests failed

### Alternative: CLI Command

The same functionality is available via the main CLI:

```bash
dicom-nlquery smoke-test
dicom-nlquery smoke-test --max-time 15
dicom-nlquery smoke-test -q "CT chest exams"
```

## run_agent.py

Interactive agent runner for testing the full agent loop with confirmation prompts.

```bash
python scripts/run_agent.py "Cranial MR for women ages 20 to 40 to RADIANT"
```

## manual_search_cli.py

Direct DICOM search without NL parsing (for debugging).

## nlquery_move_study.py

Utility for executing C-MOVE operations with the NL query pipeline.
