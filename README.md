# SepsisGuard-X

AI-based sepsis risk prediction research prototype.

## Run the application
```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/

## Run tests
```powershell
python -m pytest tests/test_api.py -v
```

## Disclaimer
Research use only. Not a clinical diagnostic tool. Do not use for patient-care decisions.

The current application uses calibrated Logistic Regression. Complete uncertainty-aware, trend-adaptive, and explainable functionality remains future work.
