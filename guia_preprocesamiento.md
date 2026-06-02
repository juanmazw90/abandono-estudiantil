## Guía de Preprocesamiento de Nuevos Datos para Inferencia

Para garantizar que el modelo XGBoost optimizado (`xgboost_optimized_model.pkl`) realice predicciones precisas sobre datos nuevos, es fundamental aplicar los mismos pasos de preprocesamiento que se utilizaron durante su entrenamiento. Esta guía detalla el proceso paso a paso:

### 1. Ingeniería de Características (Feature Engineering)

Deberás recrear las características derivadas que se construyeron a partir de los datasets originales (`studentInfo`, `studentRegistration`, `studentAssessment`, `assessments`, `vle`, `studentVle`, `courses`). Las características clave a generar incluyen:

*   **`days_active_vle`**: Calcula el número de días que un estudiante estuvo activo en el VLE. Esto implica fusionar `studentVle` con `vle`, y luego agrupar por estudiante, módulo y presentación para encontrar la diferencia entre la fecha máxima y mínima de interacción.
*   **`sum_weighted_assessment_score` / `avg_score` / `max_score` / `num_assessments`**: Estas características provienen de la fusión y agregación de los datos de `studentAssessment` y `assessments`. Implican calcular las puntuaciones promedio, máximas, sumas ponderadas y el conteo de evaluaciones por estudiante, módulo y presentación.
*   **`total_clicks_vle` / `avg_clicks_per_interaction_vle` / `num_vle_interactions` / `num_unique_vle_items`**: Estas se derivan de la agregación de `studentVle`, capturando el total y promedio de clics, el número de interacciones y de ítems únicos visitados.
*   **`vle_activity_count_X`**: Las características `vle_activity_count_dataplus`, `vle_activity_count_dualpane`, etc., se crean a partir de contar la frecuencia de cada `activity_type` dentro de `df_student_vle_enriched` (resultado de la fusión de `studentVle` y `vle`) por estudiante, módulo y presentación.
*   **`module_presentation_length_x`**: Se obtuvo de `df_courses` e indica la duración del módulo/presentación.
*   Asegúrate de que `date_registration` y `num_of_prev_attempts` estén presentes y en el formato correcto.

### 2. Manejo de Valores Faltantes (Missing Values)

Aplica las mismas estrategias de imputación utilizadas en el entrenamiento:

*   **`imd_band`**: Imputar los valores faltantes con la **moda** de la columna calculada del conjunto de entrenamiento.
*   **`date_registration`**: Imputar los valores faltantes con la **mediana** de la columna calculada del conjunto de entrenamiento.
*   **Otras características numéricas (puntuaciones, clics VLE, etc.)**: Imputar los valores faltantes con **cero (0)**. Esto incluye `avg_score`, `max_score`, `min_score`, `num_assessments`, `sum_weighted_assessment_score`, `total_clicks_vle`, `avg_clicks_per_interaction_vle`, `num_vle_interactions`, `num_unique_vle_items`, `days_active_vle`, y todas las columnas `vle_activity_count_X`.
*   **Eliminar `date_unregistration`**: Esta columna fue eliminada para evitar data leakage, ya que su información está directamente relacionada con la variable objetivo `dropout`.

### 3. Codificación One-Hot (One-Hot Encoding)

Las variables categóricas deben ser transformadas utilizando One-Hot Encoding. Es crucial que los nuevos datos utilicen los mismos *categorías* y generen las mismas *columnas dummy* que se obtuvieron del conjunto de entrenamiento. Las columnas a codificar son:

*   `code_module`
*   `code_presentation`
*   `gender`
*   `region`
*   `highest_education`
*   `imd_band`
*   `age_band`
*   `disability`

Utiliza `pd.get_dummies()` con `drop_first=True` para evitar la multicolinealidad, tal como se hizo en el entrenamiento.

### 4. Selección de Características (Feature Selection)

El modelo optimizado (`xgboost_optimized_model.pkl`) fue entrenado únicamente con las **20 características más importantes**. Debes asegurarte de que tus datos preprocesados contengan *solo estas 20 características* y que estén en el *orden exacto* especificado en el archivo `/content/poc_feature_order.json`.

*   Carga el archivo `/content/poc_feature_order.json` para obtener la lista `feature_names`.
*   Filtra tu DataFrame de nuevos datos para incluir solo estas columnas, en el orden correcto.

### 5. Escalado de Características (Scaling)

Las características numéricas (no binarias) deben ser escaladas utilizando el mismo `StandardScaler` que se ajustó con los datos de entrenamiento.

*   **Carga el `StandardScaler` previamente guardado.** (Si no se guardó, deberías ajustarlo y guardarlo en el entrenamiento original).
*   Aplica `scaler.transform()` a las columnas numéricas relevantes de tus nuevos datos.
*   Las columnas numéricas escaladas son: `num_of_prev_attempts`, `studied_credits`, `date_registration`, `avg_score`, `max_score`, `min_score`, `num_assessments`, `sum_weighted_assessment_score`, `module_presentation_length_x`, `total_clicks_vle`, `avg_clicks_per_interaction_vle`, `num_vle_interactions`, `num_unique_vle_items`, `days_active_vle`.

### Verificación Final

Después de completar estos pasos, tu DataFrame de nuevos datos debe tener:

*   Las **mismas 20 columnas** que el `X_test_optimized` o `X_train_optimized` del entrenamiento.
*   Las columnas en el **mismo orden** que `poc_feature_order.json`.
*   Los valores numéricos **escalados** y las categóricas **codificadas** correctamente.
*   **Ningún valor faltante**.

Una vez que los datos nuevos estén correctamente preprocesados, podrás cargarlos en el modelo `xgboost_optimized_model.pkl` para realizar predicciones.