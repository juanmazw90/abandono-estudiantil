# Predictor de Abandono Estudiantil

Aplicación Streamlit para inferencia de un modelo XGBoost que predice la probabilidad de abandono estudiantil, basado en el dataset OULAD (Open University Learning Analytics Dataset).

**Demo desplegada:** [abandono-estudiantil-poc.streamlit.app](https://abandono-estudiantil-poc.streamlit.app/)

## Arquitectura

```
Streamlit (app.py)
    └── Preprocesamiento (escalado con StandardScaler)
        └── Modelo XGBoost (xgboost_optimized_model.pkl)
```

## Requisitos

- [uv](https://docs.astral.sh/uv/) (gestor de dependencias)
- Python 3.12+

## Instalación y ejecución local

```bash
# Instalar dependencias
uv sync

# Ejecutar la app
uv run streamlit run app.py
```

La app estará disponible en `http://localhost:8501`.

## Artefactos del modelo

Ubicados en `model/`:

| Archivo | Descripción |
|---|---|
| `xgboost_optimized_model.pkl` | Modelo XGBoost optimizado (20 features) |
| `scaler.pkl` | StandardScaler ajustado sobre datos de entrenamiento |
| `feature_order.json` | Orden exacto de columnas para inferencia |
| `feature_meanings.json` | Descripciones de cada feature |

## Niveles de riesgo

| Probabilidad | Nivel |
|---|---|
| 0% – 39% | Bajo |
| 40% – 69% | Medio |
| 70% – 100% | Alto |
