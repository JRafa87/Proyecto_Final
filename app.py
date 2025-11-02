import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# ==========================
# 1. Cargar Modelos y Artefactos
# ==========================
@st.cache_resource
def load_model():
    try:
        # Cargar modelos y artefactos
        model = joblib.load('models/xgboost_model.pkl')
        categorical_mapping = joblib.load('models/categorical_mapping.pkl')  # Cargar el diccionario de codificación

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
        df_reference_features = df_reference.drop(columns=['Attrition']).copy()
        
        st.success("✅ Modelo cargado correctamente.")
        return model, categorical_mapping, scaler, df_reference_features, true_labels_reference

    except FileNotFoundError:
        st.error("Error: Archivos del modelo (xgboost_model.pkl, categorical_mapping.pkl, scaler.pkl) no encontrados.")
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
    missing_columns = set(model_columns) - set(df_processed.columns)
    if missing_columns:
        st.error(f"❌ Faltan columnas requeridas: {', '.join(missing_columns)}")
        return None

    df_processed = df_processed[model_columns].copy()

    # Rellenar nulos y codificar
    df_processed = df_processed.drop_duplicates()
    numeric_cols_for_fillna = df_processed.select_dtypes(include=np.number).columns.tolist()
    df_processed[numeric_cols_for_fillna] = df_processed[numeric_cols_for_fillna].fillna(df_processed[numeric_cols_for_fillna].mean())

    nominal_categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

    for col in nominal_categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).str.strip().str.upper()

            if col in categorical_mapping:
                df_processed[col] = df_processed[col].map(categorical_mapping[col])

            df_processed[col] = df_processed[col].fillna(categorical_mapping.get(col, {}).get('DESCONOCIDO', -1))

    # Escalado
    df_to_scale = df_processed[model_columns]
    try:
        df_processed[model_columns] = scaler.transform(df_to_scale)
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}")
        return None

    return df_processed


# ============================
# 3. Simulaciones: Monte Carlo y What-If
# ============================
def monte_carlo_simulation(df_features, n_simulations=100, perturbation_range=(0.95, 1.05)):
    simulations = []
    key_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany']
    
    for i in range(n_simulations):
        df_sim = df_features.copy()
        for col in key_cols:
            if col in df_sim.columns:
                perturbation_factor = np.random.uniform(perturbation_range[0], perturbation_range[1], len(df_sim))
                df_sim[col] = df_sim[col] * perturbation_factor
        simulations.append(df_sim)
    return simulations


def what_if_simulation(df_features, perturbation_factor=1.10):
    df_sim = df_features.copy()
    if 'MonthlyIncome' in df_sim.columns:
        df_sim['MonthlyIncome'] *= perturbation_factor
    return [df_sim]


# ============================
# 4. Evaluación de Simulaciones
# ============================
def evaluate_simulations(simulated_datasets, true_labels_reference, model, categorical_mapping, scaler, model_feature_columns):
    scores = []
    f1_scores = []
    true_labels = true_labels_reference.values.astype(int)

    for sim_data in simulated_datasets:
        sim_data_processed = preprocess_data(sim_data, model_feature_columns, categorical_mapping, scaler)
        if sim_data_processed is None:
            st.warning("Preprocesamiento fallido en una simulación.")
            return [], []

        probabilidad_renuncia = model.predict_proba(sim_data_processed)[:, 1]
        predictions = (probabilidad_renuncia > 0.5).astype(int)

        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        scores.append(acc)
        f1_scores.append(f1)

    return scores, f1_scores


# ============================
# 5. Exportar Resultados a Excel
# ============================
def export_results_to_excel(df, simulated_datasets=None, filename="simulation_results.xlsx"):
    with pd.ExcelWriter('temp.xlsx', engine='xlsxwriter') as output:
        # Guardar los resultados predichos
        df.to_excel(output, sheet_name='Resultados Predicción', index=False)

        # Si se tienen simulaciones, agregar cada una en una hoja separada
        if simulated_datasets:
            for i, sim_data in enumerate(simulated_datasets):
                sim_data['Simulacion_ID'] = i + 1  # Columna para identificar la simulación
                sim_data.to_excel(output, sheet_name=f'Simulacion_{i+1}', index=False)

        # También podemos agregar el detalle de la deserción (por empleado)
        df_desercion = df[['EmployeeNumber', 'Prediction_Renuncia', 'Probabilidad_Renuncia']].copy()
        df_desercion['Deserción'] = df_desercion['Prediction_Renuncia'].apply(lambda x: 'Sí' if x == 1 else 'No')
        df_desercion.to_excel(output, sheet_name='Deserción Detallada', index=False)

    # Leer el archivo para permitir la descarga
    with open('temp.xlsx', 'rb') as f:
        data = f.read()
    
    os.remove('temp.xlsx')
    return data


# ============================
# 6. Función para Graficar
# ============================
def plot_metrics(simulated_scores, simulated_f1):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].hist(simulated_scores, bins=10, color='skyblue', edgecolor='black')
    ax[0].set_title('Distribución de Accuracy (Robustez)')
    ax[0].set_xlabel('Accuracy')
    ax[0].set_ylabel('Frecuencia')
    ax[0].axvline(np.mean(simulated_scores), color='red', linestyle='dashed', linewidth=1, label=f'Media: {np.mean(simulated_scores):.4f}')
    ax[0].legend()

    ax[1].hist(simulated_f1, bins=10, color='lightcoral', edgecolor='black')
    ax[1].set_title('Distribución de F1-score (Robustez)')
    ax[1].set_xlabel('F1-score')
    ax[1].set_ylabel('Frecuencia')
    ax[1].axvline(np.mean(simulated_f1), color='red', linestyle='dashed', linewidth=1, label=f'Media: {np.mean(simulated_f1):.4f}')
    ax[1].legend()

    plt.tight_layout()
    st.pyplot(fig)


# ============================
# 7. Interfaz Streamlit
# ============================
def main():
    st.set_page_config(page_title="Predicción y Simulación de Renuncia", layout="wide")
    st.title("📊 Modelo de Predicción y Simulación de Renuncia de Empleados")
    st.markdown("Carga tu archivo de datos para obtener predicciones. Las simulaciones usan una **data de referencia** cargada en el servidor para evaluación.")

    model, categorical_mapping, scaler, df_reference_features, true_labels_reference = load_model()
    if model is None:
        return 

    model_feature_columns = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
        'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager',
        'IntencionPermanencia', 'CargaLaboralPercibida', 'SatisfaccionSalarial', 
        'ConfianzaEmpresa', 'NumeroTardanzas', 'NumeroFaltas' 
    ]
    
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

            probabilidad_renuncia = model.predict_proba(processed_df)[:, 1]
            predictions = (probabilidad_renuncia > 0.5).astype(int)
            
            df_original['Prediction_Renuncia'] = predictions
            df_original['Probabilidad_Renuncia'] = probabilidad_renuncia

            if 'Attrition' in df_original.columns:
                true_labels_uploaded = df_original['Attrition'].replace({'Yes': 1, 'No': 0}).astype(int)
                acc = accuracy_score(true_labels_uploaded, predictions)
                f1 = f1_score(true_labels_uploaded, predictions)
                st.success("✅ Predicción y Evaluación de datos cargados Completadas!")
                st.metric(label="Accuracy (Datos Cargados)", value=f"{acc:.4f}")
                st.metric(label="F1-score (Datos Cargados)", value=f"{f1:.4f}")
            else:
                st.warning("⚠️ El archivo cargado no tiene la columna 'Attrition'. Solo se muestran las predicciones.")
                
            st.download_button(
                label="⬇️ Descargar Resultados de Predicción (Excel)",
                data=export_results_to_excel(df_original),
                file_name="predicciones_resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # --- Simulaciones ---
    st.divider()
    st.header("2. Análisis de Simulaciones (Robustez y Escenarios)")

    simulation_option = st.radio("Selecciona tipo de simulación:", ["Monte Carlo", "What-If"])

    if simulation_option == "Monte Carlo":
        if st.button("▶️ Ejecutar Simulación Monte Carlo (100 Repeticiones)"):
            st.info("Simulando Monte Carlo sobre la data de referencia (perturbación aleatoria en Edad, Ingresos, Antigüedad)...")
            simulations = monte_carlo_simulation(df_reference_features)
            
            simulated_scores, simulated_f1 = evaluate_simulations(
                simulations, true_labels_reference, model, categorical_mapping, scaler, model_feature_columns
            )

            if simulated_scores:
                st.success("🎉 Simulación Monte Carlo Completada.")
                st.markdown(f"**Robustez - Accuracy Media:** `{np.mean(simulated_scores):.4f}`")
                st.markdown(f"**Robustez - F1-score Media:** `{np.mean(simulated_f1):.4f}`")
                plot_metrics(simulated_scores, simulated_f1)
                
    elif simulation_option == "What-If":
        st.markdown("Simula el impacto de un **aumento salarial del 10%** en la predicción de renuncia sobre la data de referencia.")
        if st.button("▶️ Ejecutar Simulación What-If (Aumento Salarial)"):
            st.info("Simulando escenario 'What-If'...")
            simulations = what_if_simulation(df_reference_features)
            
            simulated_scores, simulated_f1 = evaluate_simulations(
                simulations, true_labels_reference, model, categorical_mapping, scaler, model_feature_columns
            )
            
            if simulated_scores:
                st.success("🎉 Simulación What-If Completada.")
                st.markdown(f"**Impacto: Accuracy con +10% Salario:** `{simulated_scores[0]:.4f}`")
                st.markdown(f"**Impacto: F1-score con +10% Salario:** `{simulated_f1[0]:.4f}`")

# ============================
# Inicio de la Aplicación
# ============================
if __name__ == "__main__":
    main()











