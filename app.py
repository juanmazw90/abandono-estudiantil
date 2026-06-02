import json
import logging
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("model")

NUMERIC_FEATURES = [
    "days_active_vle",
    "num_assessments",
    "sum_weighted_assessment_score",
    "avg_score",
    "total_clicks_vle",
    "max_score",
    "num_vle_interactions",
    "studied_credits",
    "avg_clicks_per_interaction_vle",
    "num_of_prev_attempts",
]

BINARY_FEATURES = [
    "code_module_GGG",
    "code_module_EEE",
    "code_presentation_2014J",
    "vle_activity_count_subpage",
    "code_module_CCC",
    "vle_activity_count_homepage",
    "code_module_DDD",
    "region_Wales",
    "vle_activity_count_externalquiz",
    "imd_band_10-20",
]

# imd_band uses a hyphen in the model but we expose it as underscore in the UI
IMD_BAND_UI_KEY = "imd_band_10_20"


@st.cache_resource
def load_artifacts():
    with open(MODEL_DIR / "xgboost_optimized_model.pkl", "rb") as f:
        model = pickle.load(f)
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_order: list[str] = json.loads((MODEL_DIR / "feature_order.json").read_text())["feature_names"]
    feature_meanings: dict[str, str] = json.loads((MODEL_DIR / "feature_meanings.json").read_text())
    metadata: dict = json.loads((MODEL_DIR / "metadata.json").read_text())
    logger.info("Artefactos cargados correctamente.")
    return model, scaler, feature_order, feature_meanings, metadata


def preprocess(raw: dict, scaler, feature_order: list[str]) -> np.ndarray:
    # Rename UI key to model key for imd_band
    raw["imd_band_10-20"] = raw.pop(IMD_BAND_UI_KEY)

    df = pd.DataFrame([raw])

    # The scaler was fitted on 14 columns; only 10 of those are in the final 20 features.
    # Apply scaling parameters manually to avoid feature-name mismatch errors.
    scaler_feature_names = list(scaler.feature_names_in_)
    for feat in NUMERIC_FEATURES:
        if feat in scaler_feature_names:
            idx = scaler_feature_names.index(feat)
            df[feat] = (df[feat] - scaler.mean_[idx]) / scaler.scale_[idx]

    # Reorder columns to match training order
    df = df[feature_order]

    return df.values


def get_risk_level(probability: float) -> tuple[str, str]:
    if probability < 0.40:
        return "Bajo", "normal"
    elif probability < 0.70:
        return "Medio", "warning"
    else:
        return "Alto", "error"


def build_numeric_input(feature: str, meanings: dict) -> float:
    label = meanings.get(feature, feature)
    return st.number_input(label=label, min_value=0.0, value=0.0, step=1.0, key=feature)


def build_binary_input(feature: str, ui_key: str, meanings: dict) -> int:
    label = meanings.get(feature, feature)
    option = st.radio(label=label, options=["No", "Sí"], horizontal=True, key=ui_key)
    return 1 if option == "Sí" else 0


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Predictor de Abandono Estudiantil",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Predictor de Abandono Estudiantil")
st.caption("Ingresá los datos del estudiante para estimar el riesgo de abandono.")

try:
    model, scaler, feature_order, meanings, metadata = load_artifacts()
except Exception as e:
    st.error(f"Error cargando artefactos del modelo: {e}")
    st.stop()

with st.expander("ℹ️ Información del modelo"):
    st.json(metadata)

st.divider()

# ── Formulario ────────────────────────────────────────────────────────────────

with st.form("prediction_form"):
    st.subheader("Datos de actividad académica y VLE")

    col1, col2 = st.columns(2)

    with col1:
        days_active_vle = build_numeric_input("days_active_vle", meanings)
        num_assessments = build_numeric_input("num_assessments", meanings)
        sum_weighted_assessment_score = build_numeric_input("sum_weighted_assessment_score", meanings)
        avg_score = build_numeric_input("avg_score", meanings)
        max_score = build_numeric_input("max_score", meanings)

    with col2:
        total_clicks_vle = build_numeric_input("total_clicks_vle", meanings)
        num_vle_interactions = build_numeric_input("num_vle_interactions", meanings)
        avg_clicks_per_interaction_vle = build_numeric_input("avg_clicks_per_interaction_vle", meanings)
        studied_credits = build_numeric_input("studied_credits", meanings)
        num_of_prev_attempts = build_numeric_input("num_of_prev_attempts", meanings)

    st.divider()
    st.subheader("Características del módulo y perfil del estudiante")

    col3, col4 = st.columns(2)

    binary_inputs: dict[str, int] = {}

    with col3:
        binary_inputs["code_module_GGG"] = build_binary_input("code_module_GGG", "code_module_GGG", meanings)
        binary_inputs["code_module_EEE"] = build_binary_input("code_module_EEE", "code_module_EEE", meanings)
        binary_inputs["code_module_CCC"] = build_binary_input("code_module_CCC", "code_module_CCC", meanings)
        binary_inputs["code_module_DDD"] = build_binary_input("code_module_DDD", "code_module_DDD", meanings)
        binary_inputs["code_presentation_2014J"] = build_binary_input(
            "code_presentation_2014J", "code_presentation_2014J", meanings
        )

    with col4:
        binary_inputs["vle_activity_count_subpage"] = build_binary_input(
            "vle_activity_count_subpage", "vle_activity_count_subpage", meanings
        )
        binary_inputs["vle_activity_count_homepage"] = build_binary_input(
            "vle_activity_count_homepage", "vle_activity_count_homepage", meanings
        )
        binary_inputs["vle_activity_count_externalquiz"] = build_binary_input(
            "vle_activity_count_externalquiz", "vle_activity_count_externalquiz", meanings
        )
        binary_inputs["region_Wales"] = build_binary_input("region_Wales", "region_Wales", meanings)
        binary_inputs[IMD_BAND_UI_KEY] = build_binary_input("imd_band_10-20", IMD_BAND_UI_KEY, meanings)

    submitted = st.form_submit_button("Predecir riesgo de abandono", type="primary", use_container_width=True)

# ── Predicción ────────────────────────────────────────────────────────────────

if submitted:
    raw_input = {
        "days_active_vle": days_active_vle,
        "num_assessments": num_assessments,
        "sum_weighted_assessment_score": sum_weighted_assessment_score,
        "avg_score": avg_score,
        "total_clicks_vle": total_clicks_vle,
        "max_score": max_score,
        "num_vle_interactions": num_vle_interactions,
        "studied_credits": studied_credits,
        "avg_clicks_per_interaction_vle": avg_clicks_per_interaction_vle,
        "num_of_prev_attempts": num_of_prev_attempts,
        **binary_inputs,
    }

    try:
        X = preprocess(raw_input, scaler, feature_order)
        proba = float(model.predict_proba(X)[0][1])
        prediction = int(model.predict(X)[0])
        risk_label, risk_type = get_risk_level(proba)

        logger.info(f"Predicción: {prediction}, probabilidad: {proba:.4f}, riesgo: {risk_label}")

        st.divider()
        st.subheader("Resultado")

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.metric(label="Probabilidad de abandono", value=f"{proba:.1%}")
            st.progress(proba)

        with col_res2:
            st.metric(label="Nivel de riesgo", value=risk_label)
            if risk_type == "normal":
                st.success("Riesgo BAJO (0% – 39%): El estudiante tiene baja probabilidad de abandonar.")
            elif risk_type == "warning":
                st.warning("Riesgo MEDIO (40% – 69%): Se recomienda seguimiento preventivo.")
            else:
                st.error("Riesgo ALTO (70% – 100%): Alta probabilidad de abandono. Intervención urgente recomendada.")

    except Exception as e:
        logger.exception("Error durante la predicción.")
        st.error(f"Error durante la predicción: {e}")
