from pydantic import BaseModel, Field
from typing import List, Dict, Union

# Re-usamos el diccionario de significados de características que ya habíamos generado
# Esto asegura consistencia en las descripciones de la API.
feature_meanings = {
    "days_active_vle": "Número de días que el estudiante estuvo activo en el Entorno Virtual de Aprendizaje (VLE).",
    "num_assessments": "Número total de evaluaciones a las que el estudiante se presentó.",
    "code_module_GGG": "Indica si el estudiante está inscrito en el módulo 'GGG' (1 si sí, 0 si no).",
    "sum_weighted_assessment_score": "Suma de las puntuaciones de las evaluaciones del estudiante, ponderadas por su peso.",
    "avg_score": "Puntuación media de todas las evaluaciones del estudiante.",
    "code_module_EEE": "Indica si el estudiante está inscrito en el módulo 'EEE' (1 si sí, 0 si no).",
    "total_clicks_vle": "Número total de clics registrados por el estudiante en el VLE.",
    "code_presentation_2014J": "Indica si el estudiante está en la presentación del curso '2014J' (1 si sí, 0 si no).",
    "max_score": "Puntuación máxima obtenida por el estudiante en una evaluación.",
    "vle_activity_count_subpage": "Número de interacciones con actividades tipo 'subpage' en el VLE.",
    "code_module_CCC": "Indica si el estudiante está inscrito en el módulo 'CCC' (1 si sí, 0 si no).",
    "vle_activity_count_homepage": "Número de interacciones con la página de inicio (homepage) del VLE.",
    "code_module_DDD": "Indica si el estudiante está inscrito en el módulo 'DDD' (1 si sí, 0 si no).",
    "num_vle_interactions": "Número total de interacciones registradas en el VLE.",
    "studied_credits": "Número de créditos que el estudiante está estudiando.",
    "region_Wales": "Indica si el estudiante reside en la región de Gales (1 si sí, 0 si no).",
    "vle_activity_count_externalquiz": "Número de interacciones con actividades tipo 'externalquiz' en el VLE.",
    "imd_band_10_20": "Indica si el estudiante pertenece a la banda de Índice de Privación Múltiple (IMD) del 10-20% más desfavorecido (1 si sí, 0 si no).",
    "avg_clicks_per_interaction_vle": "Número promedio de clics por interacción del estudiante en el VLE.",
    "num_of_prev_attempts": "Número de intentos previos del estudiante para el mismo módulo/presentación."
}

class FeatureMetadata(BaseModel):
    name: str = Field(..., description="Nombre de la característica.")
    description: str = Field(..., description="Descripción del significado de la característica.")

class PredictionInput(BaseModel):
    # Características numéricas (pueden ser float después del escalado)
    days_active_vle: float = Field(..., description=feature_meanings['days_active_vle'])
    num_assessments: float = Field(..., description=feature_meanings['num_assessments'])
    sum_weighted_assessment_score: float = Field(..., description=feature_meanings['sum_weighted_assessment_score'])
    avg_score: float = Field(..., description=feature_meanings['avg_score'])
    total_clicks_vle: float = Field(..., description=feature_meanings['total_clicks_vle'])
    max_score: float = Field(..., description=feature_meanings['max_score'])
    num_vle_interactions: float = Field(..., description=feature_meanings['num_vle_interactions'])
    studied_credits: float = Field(..., description=feature_meanings['studied_credits'])
    avg_clicks_per_interaction_vle: float = Field(..., description=feature_meanings['avg_clicks_per_interaction_vle'])
    num_of_prev_attempts: float = Field(..., description=feature_meanings['num_of_prev_attempts'])

    # Características binarias (0 o 1)
    code_module_GGG: int = Field(..., description=feature_meanings['code_module_GGG'], ge=0, le=1)
    code_module_EEE: int = Field(..., description=feature_meanings['code_module_EEE'], ge=0, le=1)
    code_presentation_2014J: int = Field(..., description=feature_meanings['code_presentation_2014J'], ge=0, le=1)
    vle_activity_count_subpage: int = Field(..., description=feature_meanings['vle_activity_count_subpage'], ge=0, le=1)
    code_module_CCC: int = Field(..., description=feature_meanings['code_module_CCC'], ge=0, le=1)
    vle_activity_count_homepage: int = Field(..., description=feature_meanings['vle_activity_count_homepage'], ge=0, le=1)
    code_module_DDD: int = Field(..., description=feature_meanings['code_module_DDD'], ge=0, le=1)
    region_Wales: int = Field(..., description=feature_meanings['region_Wales'], ge=0, le=1)
    vle_activity_count_externalquiz: int = Field(..., description=feature_meanings['vle_activity_count_externalquiz'], ge=0, le=1)
    imd_band_10_20: int = Field(..., description=feature_meanings['imd_band_10_20'], ge=0, le=1)

class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="La predicción de abandono (1 para abandono, 0 para no abandono).")
    probability: float = Field(..., description="La probabilidad de abandono predicha por el modelo.")
    message: str = Field(..., description="Mensaje adicional sobre la predicción.")
