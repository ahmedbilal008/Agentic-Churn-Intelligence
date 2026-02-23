# Interview Notes — Agentic Churn Intelligence Platform

Design decisions, trade-offs, and talking points organized by component.

---

## Architecture

**Hexagonal (Ports & Adapters)**
- `src/core/` — pure ML logic with no framework dependencies
- `src/services/` — pipeline orchestration
- `src/interfaces/` — MCP server (the adapter)
- `src/schemas/` — shared data contracts (Pydantic)

Benefit: swapping the MCP interface for FastAPI or gRPC requires touching only `src/interfaces/`, not the ML logic.

**MCP (Model Context Protocol)**
- AI agents call tools over SSE transport — no custom API needed
- Server is a thin wrapper; business logic stays in `core/`
- The same `predict_churn` logic could be exposed via REST with zero changes to `core/`

---

## Data Layer

**Preprocessing consistency (train/inference skew)**
- Preprocessor is fit on training data and serialized alongside the model
- Inference always uses the same fitted preprocessor — same encoding categories, same scaling parameters, same feature ordering
- If you refit the preprocessor on new data, the feature scale changes and predictions break

**ColumnTransformer**
- Applies `StandardScaler` to numerical columns and `OneHotEncoder` to categorical columns in a single serializable pipeline
- `drop="first"` avoids multicollinearity in logistic regression; `handle_unknown="ignore"` prevents hard failures on unseen categories at inference

**Data validation**
- Pydantic schemas validate every ingested row at the pipeline boundary — catches type errors, out-of-range values, and invalid categories before they reach the model
- For millions of rows: validate a sample + enforce constraints at the SQL/storage layer instead

**Data loading separation**
- Data loading is its own module so data sources (CSV, S3, database) can be swapped without touching training or inference code

---

## Feature Engineering

**Row-level transforms only**
- All features in `engineer.py` use only per-row data (no global statistics)
- Safe to compute before the train/test split — no data leakage
- Global transforms (scaling, encoding) happen in the preprocessor after the split

**Features created:**
- `tenure_group` — categorical loyalty tier (0-12, 12-24, 24-48, 48-60, 60+); churn patterns differ sharply by segment
- `avg_charges_per_month` — TotalCharges / tenure; reveals overpaying relative to loyalty
- `charges_ratio` — MonthlyCharges / TotalCharges; high ratio = new customer spending heavily
- `num_services` — count of active services; more services = higher switching cost = lower churn
- `has_internet` — binary simplification of the 3-value InternetService column
- `senior_high_charges` — interaction: seniors paying >$70/month churn at disproportionately higher rates

---

## Model Training

**Why three models?**

| Model | Strength | When to use |
|---|---|---|
| Logistic Regression | Interpretable, fast, linear | Regulatory/audit requirements; always a useful baseline |
| Random Forest | Non-linear, robust, low variance | Good default for tabular data |
| XGBoost | State-of-the-art tabular performance | When you need max accuracy and can afford complexity |

Start with logistic regression. If it matches XGBoost, ship the simpler model — 10x faster inference, easier to debug.

**Model selection metric: F1**
- Churn prediction requires balancing precision (don't waste retention offers on loyal customers) and recall (don't miss actual churners)
- F1 = harmonic mean of precision and recall; better than accuracy on an imbalanced dataset

**MLflow tracking**
- Every run logs: hyperparameters, metrics, confusion matrix, ROC curve, feature importance plot, model artifact
- Makes experiment comparison instant and results fully reproducible
- Alternatives: Weights & Biases (better UI, cloud-first), Neptune.ai

**Model registry pattern**
- Lightweight local registry complementing MLflow: tracks which model is currently serving, its metrics, and when it was trained
- Production pattern: MLflow Model Registry with stage transitions (Staging → Production → Archived); every promotion requires metric thresholds

---

## Serving & MCP Tools

**10 MCP tools exposed:**
1. `predict_churn` — probability + risk level + top SHAP drivers (all params optional, dataset defaults applied)
2. `explain_prediction` — full SHAP waterfall explanation (all params optional)
3. `get_model_metrics` — metrics for a specific model from MLflow
4. `compare_models` — F1 leaderboard across all runs
5. `get_dataset_summary` — shape, churn rate, distributions
6. `get_feature_importance` — ranked feature importances from best model
7. `retrain_model` — trigger full pipeline retraining (threading.Lock prevents concurrent runs)
8. `add_customer_record` — append a new row to the training CSV
9. `get_active_model_info` — metadata for the currently loaded model
10. `system_status` — health check: model, preprocessor, data file, MLflow

**Retrain safety**
- `threading.Lock` prevents two simultaneous retrains
- `_ModelCache.reload()` clears the in-memory cache after retraining so the next tool call loads the new model

**Dataset defaults for predict/explain**
- All 19 input features are optional; unspecified ones fall back to the dataset mode (categoricals) or median (numericals)
- Allows the AI agent to call `predict_churn(tenure=2, Contract="Month-to-month")` without knowing all 19 fields

---

## Pipeline & Reproducibility

**DVC pipeline (4 stages)**
1. `ingest` — load raw CSV, clean, validate
2. `features` — engineer derived features
3. `train` — train all three models, log to MLflow
4. `evaluate` — select best model, save artifacts

DVC only re-runs a stage when its dependencies (code or data) change. Changing `n_estimators` only reruns `train` and `evaluate`, not `ingest`.

**params.yaml vs environment variables**
- ML hyperparameters → `params.yaml` (DVC-tracked, reproducible)
- Deployment config (port, URIs, paths) → environment variables (12-Factor App)
- Same Docker image runs in dev, staging, and production with only env vars changed

---

## Deployment Notes

**Heroku (backend) — ephemeral filesystem**
- `retrain_model`: retrains and saves `.pkl` to Heroku's filesystem. Files are lost on dyno restart. For persistence, push artifacts to S3 after retraining.
- `add_customer_record`: appends to `data/raw/churn.csv` on the dyno. Same caveat — data is lost on restart. For persistence, use a Postgres addon or S3.
- For portfolio/demo purposes, both tools work fine without persistence.

**Vercel (frontend) vs Heroku for frontend**
- Vercel is purpose-built for Next.js: instant deploys, automatic edge caching, zero config
- Heroku can host Next.js but requires manual buildpack setup and custom routing
- Vercel + Heroku (backend only) is the cleaner split

**CORS**
- The MCP server must allow Vercel's domain; set `MCP_HOST=0.0.0.0` and configure FastMCP's CORS settings or add a middleware layer

---

## Logging

- Structured format: `timestamp | level | module | message`
- In production: pipe to ELK Stack, Datadog, or CloudWatch
- Current setup uses Python's built-in `logging` — easy to swap by changing the handler in `src/utils/logger.py`

---

## Testing

- `tests/test_model.py` — unit tests for training pipeline, prediction, defaults behavior
- `tests/test_schemas.py` — Pydantic validation tests
- 24/24 tests passing
