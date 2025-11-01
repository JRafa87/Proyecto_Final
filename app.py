import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ==========================
# 1. Cargar Modelos y Artefactos
# ==========================
@st.cache_resource 
def load_model():
    """
    Carga el modelo entrenado, el label encoder, el scaler y la data de referencia.
    """
    try:
        model = joblib.load('models/xgboost_model.pkl')
        le = joblib.load('models/label_encoder.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        REFERENCE_DATA_PATH = 'data/reference_data.csv'
        if not os.path.exists(REFERENCE_DATA_PATH):
            st.error(f"Error: No se encontró la data de referencia en '{REFERENCE_DATA_PATH}'. Necesaria para evaluación de simulaciones.")
            return None, None, None, None, None
             
        df_reference = pd.read_csv(REFERENCE_DATA_PATH)
        
        if 'Attrition' not in df_reference.columns:
            st.error("Error: La data de referencia debe contener la columna 'Attrition' para la evaluación.")
            return None, None, None, None, None
            
        # Codificación de Attrition para evaluación (solución de error anterior)
        df_reference['Attrition'] = df_reference['Attrition'].replace({'Yes': 1, 'No': 0})
            
        true_labels_reference = df_reference['Attrition'].astype(int).copy()
        df_reference_features = df_reference.drop(columns=['Attrition']).copy()

        return model, le, scaler, df_reference_features, true_labels_reference
        
    except FileNotFoundError:
        st.error("Error: Archivos del modelo (xgboost_model.pkl, label_encoder.pkl, scaler.pkl) no encontrados. Asegúrate de que están en la carpeta 'models'.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error al cargar artefactos o data de referencia: {e}")
        return None, None, None, None, None


# ================================
# 2. Funciones de Preprocesamiento
# ================================
def preprocess_data(df, model_columns, le, scaler):
    """
    Preprocesa los datos cargados (sin la data de referencia).
    """
    df_processed = df.copy()
    
    # 1. Validación y filtrado de columnas
    missing_columns = set(model_columns) - set(df_processed.columns)
    if missing_columns:
        st.error(f"❌ Error de datos: Faltan las siguientes columnas requeridas por el modelo: {', '.join(missing_columns)}")
        return None

    df_processed = df_processed[[col for col in model_columns if col in df_processed.columns]]

    # 2. Eliminación de duplicados y rellenado de nulos
    df_processed = df_processed.drop_duplicates()
    df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))

    # 3. Codificación de variables categóricas
    categorical_cols = ['Gender', 'BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'OverTime']
    for col in categorical_cols:
        if col in df_processed.columns:
            try:
                # Normalizar la cadena antes de aplicar el transform (en minúsculas y quitar espacios)
                df_processed[col] = df_processed[col].astype(str).str.strip().str.lower()
                
                # Aplicar el LabelEncoder entrenado
                df_processed[col] = le.transform(df_processed[col])

            except ValueError as e:
                st.error(f"Error en la codificación de la columna '{col}'. Asegúrate de que todos los valores categóricos están presentes en el LabelEncoder. Error: {e}")
                return None
    
    # 4. Escalado de las variables numéricas
    numeric_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'JobLevel', 'MonthlyIncome', 'MonthlyRate', 
                       'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
                       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
                       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    
    cols_to_scale = [col for col in numeric_columns if col in df_processed.columns]
    
    try:
        df_processed[cols_to_scale] = scaler.transform(df_processed[cols_to_scale])
    except Exception as e:
        st.error(f"Error durante el escalado de datos: {e}")
        return None

    return df_processed


def preprocess_simulation_data(df, model_columns, le, scaler, reference_data):
    """
    Preprocesa los datos de simulación usando la data de referencia.
    """
    df_processed = df.copy()
    
    # 1. Validación y filtrado de columnas
    missing_columns = set(model_columns) - set(df_processed.columns)
    if missing_columns:
        st.error(f"❌ Error de datos: Faltan las siguientes columnas requeridas por el modelo: {', '.join(missing_columns)}")
        return None

    df_processed = df_processed[[col for col in model_columns if col in df_processed.columns]]

    # 2. Eliminación de duplicados y rellenado de nulos
    df_processed = df_processed.drop_duplicates()
    df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))

    # 3. Codificación de variables categóricas usando la data de referencia
    categorical_cols = ['Gender', 'BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'OverTime']
    for col in categorical_cols:
        if col in df_processed.columns:
            try:
                # Normalizar la cadena antes de aplicar el transform (en minúsculas y quitar espacios)
                df_processed[col] = df_processed[col].astype(str).str.strip().str.lower()
                
                # Usar la data de referencia para validar las categorías
                valid_categories = reference_data[col].astype(str).str.strip().str.lower().unique()
                df_processed[col] = df_processed[col].apply(lambda x: x if x in valid_categories else 'unknown')
                
                # Aplicar el LabelEncoder entrenado
                df_processed[col] = le.transform(df_processed[col])

            except ValueError as e:
                st.error(f"Error en la codificación de la columna '{col}'. Asegúrate de que todos los valores categóricos están presentes en el LabelEncoder. Error: {e}")
                return None
    
    # 4. Escalado de las variables numéricas
    numeric_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'JobLevel', 'MonthlyIncome', 'MonthlyRate', 
                       'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
                       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
                       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    
    cols_to_scale = [col for col in numeric_columns if col in df_processed.columns]
    
    try:
        df_processed[cols_to_scale] = scaler.transform(df_processed[cols_to_scale])
    except Exception as e:
        st.error(f"Error durante el escalado de datos: {e}")
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
def evaluate_simulations(simulated_datasets, true_labels_reference, model, le, scaler, model_feature_columns):
    """
    Evalúa el rendimiento de las simulaciones comparando las predicciones
    con las etiquetas verdaderas de la data de REFERENCIA.
    """
    scores = []
    f1_scores = []
    
    true_labels = true_labels_reference.values.astype(int) 

    for sim_data in simulated_datasets:
        # La data simulada ya contiene solo FEATURES, se preprocesa directamente
        sim_data_processed = preprocess_simulation_data(sim_data, model_feature_columns, le, scaler, reference_data)
        
        if sim_data_processed is None:
            st.warning("Preprocesamiento fallido en una simulación. Se detiene la evaluación.")
            return [], [] 
        
        # Predicción
        probabilidad_renuncia = model.predict_proba(sim_data_processed)[:, 1]
        predictions = (probabilidad_renuncia > 0.5).astype(int)
        
        # Evaluación: Predicciones de la simulación vs. Etiquetas de REFERENCIA
        try:
            if len(predictions) != len(true_labels):
                st.error(f"Error de simulación: El número de filas simuladas ({len(predictions)}) no coincide con las etiquetas de referencia ({len(true_labels)}).")
                return [], []
                
            acc = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions)
            
            scores.append(acc)
            f1_scores.append(f1)
            
        except Exception as e:
            st.error(f"Error al evaluar la simulación: {e}")
            return [], []

    return scores, f1_scores


# ============================
# 5. Exportar Resultados a Excel
# ============================
def export_results_to_excel(df, filename="simulation_results.xlsx"):
    output = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')
    df.to_excel(output, sheet_name='Resultados', index=False)
    output.close()
    
    with open('temp.xlsx', 'rb') as f:
        data = f.read()
    
    os.remove('temp.xlsx')
    
    return data


# ============================
# 6. Función para Graficar Métricas
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
# 7. Interfaz de Streamlit
# ============================
def main():
    st.set_page_config(page_title="Predicción y Simulación de Renuncia", layout="wide")
    st.title("📊 Modelo de Predicción y Simulación de Renuncia de Empleados")
    st.markdown("Carga tu archivo de datos para obtener predicciones. Las simulaciones usan una **data de referencia** cargada en el servidor para evaluación.")

    model, le, scaler, df_reference_features, true_labels_reference = load_model()
    if model is None:
        return 

    model_feature_columns = list(df_reference_features.columns)

    # --- Columna para la carga de archivos y Predicción ---
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
        processed_df = preprocess_data(df_features_uploaded, model_feature_columns, le, scaler)
        
        if processed_df is None:
            st.error("No se puede continuar con la predicción debido a un error de preprocesamiento en el archivo cargado.")
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

    # --- Separador y Opciones de Simulación ---
    st.divider()
    st.header("2. Análisis de Simulaciones (Robustez y Escenarios)")
    st.markdown("Las simulaciones se basan en el **dataset de referencia** para garantizar la evaluación.")
    
    simulation_option = st.radio("Selecciona tipo de simulación:", ["Monte Carlo", "What-If"])

    if simulation_option == "Monte Carlo":
        if st.button("▶️ Ejecutar Simulación Monte Carlo (100 Repeticiones)"):

            # Simulación Monte Carlo
            st.info("Simulando Monte Carlo sobre la data de referencia...")
            
            simulations = monte_carlo_simulation(df_reference_features)
            
            simulated_scores, simulated_f1 = evaluate_simulations(
                simulations, true_labels_reference, model, le, scaler, model_feature_columns
            )

            if simulated_scores:
                st.success("🎉 Simulación Monte Carlo Completada.")
                st.markdown(f"**Robustez - Accuracy Media:** `{np.mean(simulated_scores):.4f}`")
                st.markdown(f"**Robustez - F1-score Media:** `{np.mean(simulated_f1):.4f}`")
                plot_metrics(simulated_scores, simulated_f1)
                
    elif simulation_option == "What-If":
        st.markdown("Simula el impacto de un **aumento salarial del 10%** en la predicción de renuncia sobre la data de referencia.")
        if st.button("▶️ Ejecutar Simulación What-If (Aumento Salarial)"):

            # Simulación What-If
            st.info("Simulando escenario 'What-If'...")

            simulations = what_if_simulation(df_reference_features)
            
            simulated_scores, simulated_f1 = evaluate_simulations(
                simulations, true_labels_reference, model, le, scaler, model_feature_columns
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



