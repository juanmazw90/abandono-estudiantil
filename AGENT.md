# Prompt revisado

## Contexto General

Actúa como un Senior Machine Learning Engineer, Backend Engineer y MLOps Engineer.

Debes construir una aplicación PoC completa para inferencia de un modelo de abandono estudiantil utilizando:

- Backend: FastAPI
- Frontend: Streamlit
- Modelo: XGBoost optimizado
- Gestión de dependencias: UV
- Contenedorización: Docker
- Control de versiones: Git/GitHub

La solución debe ser completamente funcional, reproducible y lista para despliegue cloud.

---

# Estructura esperada del proyecto

```text
project/
│
├── backend/
│   ├── main.py
│   ├── data_models.py
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── Dockerfile
│   │
│   └── model/
│       ├── model.joblib
│       ├── scaler.joblib
│       ├── feature_order.json
│       ├── feature_meanings.json
│       ├── feature_meanings.json
│       └── metadata.json
│
├── frontend/
│   ├── app.py
│   ├── pyproject.toml
│   ├── uv.lock
│   └── Dockerfile
│
├── docker-compose.yml
│
├── guia_preprocesamiento.md
│
└── README.md
```

## Gestión de dependencias

Utiliza UV como gestor oficial de dependencias y entornos.

NO utilices requirements.txt salvo que una plataforma de despliegue lo requiera explícitamente.

Trabaja mediante:

```bash
uv init
uv add ...
```

y versiona:

```text
pyproject.toml
uv.lock
```

Las imágenes Docker deberán utilizar UV para instalar dependencias.

## Desarrollo del Backend FastAPI

Crear:

```text
backend/main.py
```

### Carga de artefactos

Al iniciar la aplicación cargar:

- model.joblib
- scaler.joblib
- feature_order.json
- feature_meanings.json
- metadata.json

desde `backend/model/`.

Los artefactos deben cargarse una única vez durante el startup de la aplicación.

### Modelos Pydantic

NO crear modelos nuevos.

Importar y utilizar exclusivamente los modelos ya definidos en:

```python
from data_models import *
```

Utilizar esos modelos para:

- validación de requests
- validación de responses
- documentación OpenAPI

### Preprocesamiento

Implementar una función de preprocesamiento que replique EXACTAMENTE el comportamiento documentado en:

`guia_preprocesamiento.md`

No simplificar ni reinterpretar la lógica.

Debe incluir, si aplica:

- limpieza de datos
- feature engineering
- imputación
- transformaciones
- normalización
- escalado
- generación de variables derivadas
- codificación de variables categóricas
- cualquier otra transformación documentada

#### Orden de columnas

La matriz final debe respetar exactamente `feature_order.json`.

#### One Hot Encoding

Las columnas dummy generadas deben coincidir exactamente con las utilizadas durante entrenamiento.

No deben aparecer columnas nuevas.

No deben faltar columnas existentes.

Las categorías desconocidas deben gestionarse de forma segura.

#### Escalado

Aplicar exclusivamente `scaler.joblib`.

No recalcular escaladores.

No volver a entrenar transformaciones.

### Endpoints

#### GET /

```json
{
  "service": "student-dropout-api",
  "status": "running"
}
```

#### GET /health

```json
{
  "status": "healthy"
}
```

#### GET /model-info

Retornar información contenida en `metadata.json`.

#### POST /predict

Debe:

1. Validar request mediante Pydantic.
2. Aplicar preprocesamiento.
3. Ejecutar inferencia.
4. Obtener probabilidad de abandono.
5. Retornar respuesta estructurada.

Ejemplo:

```json
{
  "probability": 0.83,
  "risk_level": "HIGH"
}
```

### Gestión de errores

Implementar manejo robusto para:

- campos faltantes
- tipos incorrectos
- categorías inválidas
- valores fuera de rango
- errores de inferencia
- errores de carga de artefactos

Retornar códigos HTTP apropiados.

### Logging

Implementar logging estructurado.

Registrar:

- inicio de aplicación
- carga de modelo
- requests recibidas
- errores
- inferencias realizadas

Sin registrar información sensible.

### CORS

Configurar CORSMiddleware permitiendo acceso desde el frontend Streamlit.

La URL deberá poder configurarse mediante variable de entorno.

## Desarrollo del Frontend Streamlit

Crear `frontend/app.py`.

### Interfaz

Solicitar únicamente las 20 features utilizadas por el modelo optimizado.

### Descripciones amigables

Utilizar `feature_meanings.json` para:

- etiquetas
- ayudas contextuales
- tooltips
- textos descriptivos

### Tipado de controles

Seleccionar automáticamente el widget adecuado:

- st.number_input
- st.slider
- st.selectbox
- st.radio
- st.checkbox

según el tipo de dato.

### Validación

Validar:

- campos obligatorios
- rangos numéricos
- categorías válidas
- formatos permitidos

antes de invocar la API.

### Comunicación con Backend

La URL del backend debe obtenerse desde:

`BACKEND_API_URL`

No hardcodear URLs.

### Inferencia

1. Construir payload JSON.
2. Invocar POST /predict.
3. Gestionar errores.
4. Mostrar resultado.

### Visualización de resultados

Probabilidad:

```text
Probabilidad de abandono: 83%
```

Nivel de riesgo:

```text
0.00 - 0.39 → Bajo
0.40 - 0.69 → Medio
0.70 - 1.00 → Alto
```

## Dockerización Backend

Crear `backend/Dockerfile`.

- Base: python:3.12-slim
- Instalar UV
- Copiar main.py, data_models.py, model/, pyproject.toml y uv.lock
- Ejecutar `uv sync --frozen`
- Exponer puerto 8000
- Ejecutar `uv run uvicorn main:app --host 0.0.0.0 --port 8000`

## Dockerización Frontend

Crear `frontend/Dockerfile`.

- Base: python:3.12-slim
- Instalar UV
- Copiar app.py, pyproject.toml y uv.lock
- Ejecutar `uv sync --frozen`
- Exponer puerto 8501
- Ejecutar `uv run streamlit run app.py`

## Docker Compose

Crear `docker-compose.yml` con:

- servicio backend
- servicio frontend
- networking interno
- variables de entorno necesarias

Permitir:

```bash
docker compose up --build
```

## GitHub

Versionar:

- código fuente
- Dockerfiles
- pyproject.toml
- uv.lock
- artefactos del modelo
- documentación

Crear `.gitignore` apropiado.

## Despliegue Cloud

### Backend

Desplegar en:

- Render
o
- Railway

Debe exponer endpoint público HTTPS.

### Frontend

Desplegar en Streamlit Community Cloud.

Configurar:

`BACKEND_API_URL`

como Secret o Variable de Entorno.

## Documentación

Generar README.md completo incluyendo:

- descripción
- instalación local
- uso de UV
- ejecución local
- Docker Compose
- variables de entorno
- despliegue
- testing

## Criterios de aceptación

- El backend inicia correctamente.
- El modelo carga correctamente.
- El preprocesamiento replica exactamente guia_preprocesamiento.md.
- Se utilizan los modelos Pydantic de data_models.py.
- La inferencia devuelve probabilidades válidas.
- El frontend consume correctamente el backend.
- Docker Compose levanta ambos servicios.
- La aplicación puede desplegarse sin modificaciones adicionales.
- El código está tipado, documentado y organizado siguiendo buenas prácticas de ingeniería de software y MLOps.
