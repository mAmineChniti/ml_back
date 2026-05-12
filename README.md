# FastAPI backend

This backend serves prediction endpoints for the three DSO flows used by the notebook and future UI.

## Endpoints

- `GET /health`
- `POST /dso1/predict`
- `POST /dso2/predict`
- `POST /dso3/predict`
- `POST /dso1/predict/batch`
- `POST /dso2/predict/batch`
- `POST /dso3/predict/batch`
- `GET /dso1/evaluation`
- `GET /dso2/evaluation`
- `GET /dso3/overview`

## Run

```bash
cd backend
/usr/bin/python -m pip install -r requirements.txt
/usr/bin/python -m uvicorn app.main:app --reload --port 8000
```

## UI integration

The evaluation endpoints return chart-ready Plotly JSON payloads so a React UI can render them directly or transform them into its own chart library.
