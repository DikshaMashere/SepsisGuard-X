# SepsisGuard-X



## AI-Based Sepsis Risk Prediction — Research Prototype



SepsisGuard-X is a machine-learning research prototype designed to estimate sepsis risk from clinical observation data. It combines calibrated risk prediction, temporal prediction, model explainability, and supporting analysis APIs in a FastAPI application with a web-based frontend.



> **Disclaimer:** This is an experimental research prototype, not a clinically validated diagnostic tool. Do not use it for diagnosis, treatment, or real patient-care decisions.



## Features



\- **Risk Prediction:** Calibrated Logistic Regression for single-observation risk estimation.

\- **Temporal Prediction:** Uses seven consecutive hourly observations to estimate risk over a six-hour prediction horizon.

\- **Model Explainability:** SHAP-based explanations of model predictions.

\- **Trend Analysis:** API functionality for examining prediction trends.

\- **Uncertainty Analysis:** Supporting uncertainty-related services and evaluation scripts.

\- **Web Interface:** Browser-based frontend integrated with the FastAPI backend.

\- **Automated Testing:** Tests for API endpoints and supporting services.



## Technology Stack



\- Python

\- FastAPI

\- Uvicorn

\- scikit-learn

\- SHAP

\- HTML, CSS, and JavaScript

\- pytest



## Project Structure



```text

SepsisGuard-X/

â”œâ”€â”€ app/             # API, schemas, and application services

â”œâ”€â”€ data/            # Local datasets and model artifacts

â”œâ”€â”€ docs/             # Documentation

â”œâ”€â”€ frontend/         # Web interface

â”œâ”€â”€ scripts/          # Data preparation and evaluation

â”œâ”€â”€ tests/            # Automated tests

â”œâ”€â”€ requirements.txt

â””â”€â”€ README.md

```



## Installation



Run the following commands in PowerShell from the project root.



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

python -m pip install -r requirements.txt

```



## Run the Application



```powershell

python -m uvicorn app.main:app --reload

```



Open the web application:



http://127.0.0.1:8000/



Interactive API documentation:



http://127.0.0.1:8000/docs



## Run Tests



Run the complete test suite:



```powershell

python -m pytest -q

```



The latest reported test run passed **18 tests**, with one deprecation warning.



## Temporal Model Evaluation



Reported held-out test results for the corrected temporal model:



| Metric | Result |

|---|---:|

| ROC-AUC | 0.7643 |

| PR-AUC | 0.0597 |

| Brier Score | 0.01391 |

| Positive Target Prevalence | 1.43% |



These metrics describe performance on the evaluated research dataset only. They do not establish clinical effectiveness, safety, or readiness for deployment.



## Data and Model Artifacts



Generated datasets and trained model artifacts are excluded from Git. The application may require local model files to run.



If required artifacts are missing, follow the available data preparation and model training scripts. A fresh clone of the repository may not run until these artifacts are created or supplied.



Never commit private, sensitive, or identifiable patient information.



## Limitations



\- The system has not been clinically validated.

\- Performance may vary across datasets and patient populations.

\- Sepsis-positive outcomes are relatively rare, so precision-recall metrics require careful interpretation.

\- Predictions and model explanations are experimental and must not guide real-world medical decisions.



## License



No open-source license has been added yet.


