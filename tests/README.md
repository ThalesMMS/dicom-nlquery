# Tests

## Unit tests

```bash
cd dicom-nlquery
pytest tests/unit/ -v
```

## Integration tests (Orthanc)

```bash
cd dicom-nlquery

docker compose -f tests/docker-compose.yml up -d
pytest tests/integration/ -v -m integration

docker compose -f tests/docker-compose.yml down
```

## Full suite

```bash
cd dicom-nlquery
pytest tests/ -v
```

## Notes

- Integration tests require Docker and free ports 4242/8042.
- If you change the Orthanc config, restart the container.
