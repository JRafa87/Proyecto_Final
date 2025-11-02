import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# ============================
# 1. Cargar Modelos y Artefactos
# ============================
@st.cache_resource
def load_model():
    try:
        # Cargar modelos y artefactos
        model = joblib.load('models/xgboost_model.pkl')
        categorical_mapping = joblib.load('models/categorical_mapping.pkl')  # Cargar el diccionario de codificación

        # Aseguramos que el scaler es el objeto correcto (MinMaxScaler o similar)
        scaler = joblib.load('models/scaler.pkl')

        REFERENCE_DATA_PATH = 'data/reference_data.csv'
        if not os.path.exists(REFERENCE_DATA_PATH):
            st.error(f"Error: No se encontró la data de referencia en '{REFERENCE_DATA_PATH}'. Necesaria para evaluación de simulaciones.")
            return None, None, None, None, None

        df_reference = pd.read_csv(REFERENCE_DATA_PATH)

        if 'Attrition' not in df_reference.columns:
            st.error("Error: La data de referencia debe contener la columna 'Attrition' para la evaluación.")
            return None, None, None, None, None

        # Solución al error 'invalid literal for int(): 'Yes''
        df_reference['Attrition'] = df_reference['Attrition'].replace({'Yes': 1, 'No': 0})

        true_labels_reference = df_reference['Attrition'].astype(int).copy()
        df_reference_features = df_reference.drop(columns=['Attrition'], errors='ignore').copy()

        st.success("✅ Modelo y artefactos cargados correctamente.")
        return model, categorical_mapping, scaler, df_reference_features, true_labels_reference

    except FileNotFoundError:
        st.error("Error: Archivos del modelo (xgboost_model.pkl, categorical_mapping.pkl, scaler.pkl) no encontrados. Asegúrate de tener la carpeta 'models' con los 3 archivos.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error al cargar artefactos o data de referencia: {e}")
        return None, None, None, None, None


# ================================
# 2. Funciones de Preprocesamiento
# ================================
def preprocess_data(df, model_columns, categorical_mapping, scaler):
    df_processed = df.copy()

    # Validar columnas
    df_processed = df_processed.drop_duplicates()
    numeric_cols_for_fillna = df_processed.select_dtypes(include=np.number).columns.tolist()

    cols_to_fill = list(set(numeric_cols_for_fillna) & set(model_columns))
    df_processed[cols_to_fill] = df_processed[cols_to_fill].fillna(df_processed[cols_to_fill].mean())

    nominal_categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    for col in nominal_categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).str.strip().str.upper()

            if col in categorical_mapping:
                df_processed[col] = df_processed[col].map(categorical_mapping[col])

            df_processed[col] = df_processed[col].fillna(categorical_mapping.get(col, {}).get('DESCONOCIDO', -1))

    df_to_scale = df_processed[model_columns].copy()

    try:
        df_processed[model_columns] = scaler.transform(df_to_scale)
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}. ¿El scaler fue entrenado correctamente con la forma de la data?")
        return None

    return df_processed[model_columns]

# ============================
# 3. Funciones de Evaluación
# ============================
def check_class_imbalance(df, target_column='Attrition'):
    class_counts = df[target_column].value_counts()
    total_instances = len(df)
    class_percentages = class_counts / total_instances * 100
    if class_percentages.min() < 30:
        return True
    else:
        return False

def dynamic_threshold(probabilidad_renuncia, imbalance_detected):
    if imbalance_detected:
        threshold = 0.3  # Reduce el umbral en datos desbalanceados
    else:
        threshold = 0.5  # Mantén el umbral estándar

    predictions = (probabilidad_renuncia > threshold).astype(int)
    return predictions

def evaluate_metrics(true_labels, predictions, imbalance_detected):
    if imbalance_detected:
        f1 = f1_score(true_labels, predictions, average='weighted')  # F1 ponderado para desbalanceo
    else:
        f1 = f1_score(true_labels, predictions, average='binary')
    acc = accuracy_score(true_labels, predictions)
    return acc, f1

# ============================
# 4. Función Principal
# ============================
def main():
    st.set_page_config(page_title="Predicción y Simulación de Renuncia", layout="wide")
    st.title("📊 Modelo de Predicción y Simulación de Renuncia de Empleados")
    st.markdown("Carga tu archivo de datos para obtener predicciones. Las simulaciones usan una **data de referencia** cargada en el servidor para evaluación.")

    model, categorical_mapping, scaler, df_reference_features, true_labels_reference = load_model()
    if model is None:
        return 

    model_feature_columns = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
                             'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                             'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                             'YearsSinceLastPromotion', 'YearsWithCurrManager', 'IntencionPermanencia', 'CargaLaboralPercibida', 'SatisfaccionSalarial', 'ConfianzaEmpresa', 'NumeroTardanzas', 'NumeroFaltas']

    uploaded_file = st.file_uploader("Sube un archivo CSV o Excel (.csv, .xlsx) para PREDICCIÓN", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.info(f"✅ Archivo cargado correctamente. Total de filas: {len(df)}")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return

        df_original = df.copy()
        df_features_uploaded = df_original.drop(columns=['Attrition'], errors='ignore').copy()

        processed_df = preprocess_data(df_features_uploaded, model_feature_columns, categorical_mapping, scaler)

        if processed_df is None:
            st.error("No se puede continuar con la predicción debido a un error de preprocesamiento.")
            return

        st.header("1. Predicción con Datos Cargados")

        if st.button("🚀 Ejecutar Predicción y Evaluación"):
            st.info("Ejecutando el modelo sobre los datos cargados...")

            imbalance_detected = check_class_imbalance(df, target_column='Attrition')

            probabilidad_renuncia = model.predict_proba(processed_df)[:, 1]
            predictions = dynamic_threshold(probabilidad_renuncia, imbalance_detected)

            df_original['Prediction_Renuncia'] = predictions
            df_original['Probabilidad_Renuncia'] = probabilidad_renuncia

            if 'Attrition' in df_original.columns:
                true_labels_uploaded = df_original['Attrition'].replace({'Yes': 1, 'No': 0}).astype(int)
                acc, f1 = evaluate_metrics(true_labels_uploaded, predictions, imbalance_detected)
                st.success("✅ Predicción y Evaluación de datos cargados Completadas!")
                st.metric(label="Accuracy (Datos Cargados)", value=f"{acc:.4f}")
                st.metric(label="F1-score (Datos Cargados)", value=f"{f1:.4f}")
            else:
                st.warning("⚠️ El archivo cargado no tiene la columna 'Attrition'. Solo se muestran las predicciones.")

            display_executive_analysis(df_original, title_suffix="Predicha")
            display_recommendations(df_original)

            st.download_button(
                label="⬇️ Descargar Resultados de Predicción (Excel)",
                data=export_results_to_excel(df_original),
                file_name="predicciones_resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()












