import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# ==========================
# 1. Cargar Modelos y Artefactos
# ==========================
@st.cache_resource # Caching para evitar recargar en cada interacción
def load_model():
    """
    Carga el modelo entrenado, el label encoder y el scaler.
    """
    try:
        # Asegúrate de que las rutas son correctas para tu proyecto
        model = joblib.load('models/xgboost_model.pkl')
        le = joblib.load('models/label_encoder.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, le, scaler
    except FileNotFoundError:
        st.error("Error: Archivos del modelo (xgboost_model.pkl, label_encoder.pkl, scaler.pkl) no encontrados. Asegúrate de que están en la carpeta 'models'.")
        return None, None, None


# ================================
# 2. Funciones de Preprocesamiento
# ================================
def preprocess_data(df, model_columns, le, scaler):
    """
    Preprocesa los datos: codificación y escalado.
    Si faltan columnas, muestra advertencia y detiene el flujo.
    """
    # 1. Copia del DataFrame para evitar SettingWithCopyWarning
    df_processed = df.copy()
    
    # 2. Eliminar columnas que no están en la lista del modelo
    # Esto debe hacerse ANTES de la validación de columnas
    cols_to_keep = [col for col in model_columns if col in df_processed.columns]
    
    # Validar si faltan columnas CRÍTICAS (las que el modelo espera)
    missing_columns = set(model_columns) - set(df_processed.columns)
    # Se ignora 'Attrition' si está en model_columns, ya que es la variable objetivo
    if 'Attrition' in missing_columns:
        missing_columns.remove('Attrition')

    if missing_columns:
        st.error(f"❌ Error de datos: Faltan las siguientes columnas requeridas por el modelo: {', '.join(missing_columns)}")
        return None  # Detener ejecución si faltan columnas

    # Reducir el DataFrame solo a las columnas relevantes
    df_processed = df_processed[cols_to_keep]

    # 3. Eliminar duplicados
    df_processed = df_processed.drop_duplicates()

    # 4. Completar los valores nulos (imputar con la media de la columna)
    df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))

    # 5. Codificación de variables categóricas
    categorical_cols = ['Gender', 'BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'OverTime']
    for col in categorical_cols:
        if col in df_processed.columns:
            try:
                # Usamos le.transform con la variable categórica
                df_processed[col] = le.transform(df_processed[col].astype(str))
            except ValueError as e:
                st.error(f"Error en la codificación de la columna '{col}'. Asegúrate de que todos los valores categóricos están presentes en el LabelEncoder. Error: {e}")
                return None
    
    # 6. Escalado de las variables numéricas
    numeric_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'JobLevel', 'MonthlyIncome', 'MonthlyRate', 
                       'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
                       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
                       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    
    # Filtrar solo las columnas numéricas presentes en el DataFrame
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
def monte_carlo_simulation(df, n_simulations=100, perturbation_range=(0.95, 1.05)):
    """
    Realiza simulaciones de Monte Carlo generando perturbaciones aleatorias sobre las variables clave.
    """
    simulations = []
    # Las columnas clave son las que se van a perturbar en la simulación
    key_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany']
    
    for i in range(n_simulations):
        df_sim = df.copy()
        for col in key_cols:
            if col in df_sim.columns:
                perturbation_factor = np.random.uniform(perturbation_range[0], perturbation_range[1], len(df_sim))
                df_sim[col] = df_sim[col] * perturbation_factor
        
        # Añadir un ID de simulación para el seguimiento
        df_sim['Simulation_ID'] = i + 1
        simulations.append(df_sim)
    return simulations


def what_if_simulation(df, perturbation_factor=1.10):
    """
    Simula escenarios 'What-If' variando un parámetro clave (ej. aumentar el salario en un 10%).
    """
    df_sim = df.copy()
    if 'MonthlyIncome' in df_sim.columns:
        df_sim['MonthlyIncome'] *= perturbation_factor  # Aumentar salario
    return [df_sim]  # Solo una simulación en este caso


# ===========================
# 4. Evaluación de Simulaciones
# ===========================
def evaluate_simulations(simulated_datasets, true_labels, model, le, scaler, model_columns):
    """
    Evalúa el rendimiento de las simulaciones calculando las métricas de precisión y F1-score.
    """
    scores = []
    f1_scores = []
    
    # Convertir a numpy array para un acceso más eficiente
    true_labels = true_labels.values.astype(int) 

    for sim_data in simulated_datasets:
        # Nota: Aquí se está preprocesando el DataFrame PERTURBADO
        # que NO tiene la columna 'Attrition' (si fue eliminada en el preprocesamiento inicial).
        # Asegúrate de que preprocess_data no dependa de 'Attrition'.
        sim_data_processed = preprocess_data(sim_data, model_columns, le, scaler)
        
        if sim_data_processed is None:
            st.warning("Preprocesamiento fallido en una simulación. Se detiene la evaluación.")
            return [], [] # Detener ejecución si preprocesamiento falla
        
        # Asegúrate de que las columnas coincidan exactamente con las que el modelo fue entrenado
        # (Esto se maneja al pasar 'model_columns' a preprocess_data)
        
        probabilidad_renuncia = model.predict_proba(sim_data_processed)[:, 1]
        predictions = (probabilidad_renuncia > 0.5).astype(int)  # Umbral de 0.5 para clasificación binaria
        
        # Las métricas se calculan comparando las predicciones de la simulación
        # contra las etiquetas VERDADERAS ORIGINALES.
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        scores.append(acc)
        f1_scores.append(f1)
    
    return scores, f1_scores


# ============================
# 5. Exportar Resultados a Excel
# ============================
def export_results_to_excel(df, filename="simulation_results.xlsx"):
    """
    Exporta los resultados de predicción a un archivo Excel.
    """
    # Crear un buffer en memoria para la descarga
    output = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')
    df.to_excel(output, sheet_name='Resultados', index=False)
    output.close()
    
    with open('temp.xlsx', 'rb') as f:
        data = f.read()
    
    # Limpiar el archivo temporal
    os.remove('temp.xlsx')
    
    return data


# ============================
# 6. Función para Graficar Métricas
# ============================
def plot_metrics(simulated_scores, simulated_f1):
    """
    Plotea las métricas de las simulaciones: Accuracy y F1-score.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Graficar Accuracy
    ax[0].hist(simulated_scores, bins=10, color='skyblue', edgecolor='black')
    ax[0].set_title('Distribución de Accuracy (Robustez)')
    ax[0].set_xlabel('Accuracy')
    ax[0].set_ylabel('Frecuencia')
    ax[0].axvline(np.mean(simulated_scores), color='red', linestyle='dashed', linewidth=1, label=f'Media: {np.mean(simulated_scores):.4f}')
    ax[0].legend()

    # Graficar F1-score
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
    st.markdown("Carga tu archivo de datos para obtener predicciones y realizar simulaciones de robustez (Monte Carlo) y escenarios (What-If).")

    # Cargar el modelo, el encoder y el scaler
    model, le, scaler = load_model()
    if model is None:
        return # Detener si no se cargan los artefactos

    # Lista de columnas que el modelo espera
    model_columns = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
        'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager', 'IntencionPermanencia',
        'CargaLaboralPercibida', 'SatisfaccionSalarial', 'ConfianzaEmpresa', 'NumeroTardanzas', 'NumeroFaltas', 
        'Attrition' # Incluimos Attrition para la evaluación, aunque se debe excluir antes de la predicción
    ]

    # --- Columna para la carga de archivos y Predicción ---
    with st.container():
        uploaded_file = st.file_uploader("Sube un archivo CSV o Excel (.csv, .xlsx)", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            # Cargar el archivo como DataFrame
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else: # Asumir xlsx
                    df = pd.read_excel(uploaded_file)
                st.info(f"✅ Archivo cargado correctamente. Total de filas: {len(df)}")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")
                return

            # Crear una copia para la exportación de resultados
            df_original = df.copy() 
            
            # 1. Preprocesamiento (Aplica los transformadores entrenados)
            # Quitar 'Attrition' de las columnas a preprocesar/predecir si está presente
            cols_for_processing = [col for col in model_columns if col != 'Attrition']
            
            # Usamos una copia de df_original para el preprocesamiento
            processed_df = preprocess_data(df_original.drop(columns=['Attrition'], errors='ignore'), cols_for_processing, le, scaler)
            
            # 2. Verificación de éxito en el preprocesamiento
            if processed_df is None:
                st.error("No se puede continuar. Por favor, verifica el mensaje de error anterior sobre las columnas faltantes o la codificación.")
                return 

            st.header("1. Predicción con Datos Cargados")
            
            # Botón para ejecutar el modelo con los datos cargados
            if st.button("🚀 Ejecutar Predicción y Evaluación"):
                st.info("Ejecutando el modelo sobre los datos cargados...")
                
                # Predicción
                probabilidad_renuncia = model.predict_proba(processed_df)[:, 1]
                predictions = (probabilidad_renuncia > 0.5).astype(int)
                
                # Añadir resultados al DataFrame original para exportación
                df_original['Prediction_Renuncia'] = predictions
                df_original['Probabilidad_Renuncia'] = probabilidad_renuncia
                
                # Evaluación
                if 'Attrition' not in df_original.columns:
                    st.warning("⚠️ La columna 'Attrition' (etiquetas verdaderas) no se encontró. No se puede calcular Accuracy ni F1-score.")
                else:
                    true_labels = df_original['Attrition'].astype(int)
                    acc = accuracy_score(true_labels, predictions)
                    f1 = f1_score(true_labels, predictions)
                    st.success("✅ Predicción y Evaluación Completadas!")
                    st.metric(label="Accuracy", value=f"{acc:.4f}")
                    st.metric(label="F1-score", value=f"{f1:.4f}")

                # Opción para descargar los resultados de la predicción
                st.download_button(
                    label="⬇️ Descargar Resultados de Predicción (Excel)",
                    data=export_results_to_excel(df_original),
                    file_name="predicciones_resultados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # --- Separador y Opciones de Simulación ---
            st.divider()
            st.header("2. Análisis de Simulaciones")
            
            # Verificar si existe la columna 'Attrition' para poder ejecutar simulaciones
            if 'Attrition' not in df_original.columns:
                st.error("🚨 Las simulaciones (Monte Carlo y What-If) requieren la columna **'Attrition'** (etiquetas verdaderas) para poder evaluar la robustez del modelo.")
                return

            simulation_option = st.radio("Selecciona tipo de simulación:", ["Monte Carlo", "What-If"])

            if simulation_option == "Monte Carlo":
                if st.button("▶️ Ejecutar Simulación Monte Carlo (100 Repeticiones)"):
                    st.info("Simulando Monte Carlo (perturbación aleatoria en Edad, Ingresos, Antigüedad)...")
                    
                    # Ejecutar y evaluar
                    simulations = monte_carlo_simulation(df_original)
                    simulated_scores, simulated_f1 = evaluate_simulations(
                        simulations, df_original['Attrition'], model, le, scaler, cols_for_processing
                    )

                    if simulated_scores:
                        st.success("🎉 Simulación Monte Carlo Completada.")
                        st.markdown(f"**Robustez - Accuracy Media:** `{np.mean(simulated_scores):.4f}`")
                        st.markdown(f"**Robustez - F1-score Media:** `{np.mean(simulated_f1):.4f}`")
                        plot_metrics(simulated_scores, simulated_f1)
                        st.warning("Nota: La exportación de resultados detallados de Monte Carlo (todas las simulaciones) no está implementada en esta versión.")
                        
            elif simulation_option == "What-If":
                st.markdown("Simula el impacto de un **aumento salarial del 10%** en la predicción de renuncia.")
                if st.button("▶️ Ejecutar Simulación What-If (Aumento Salarial)"):
                    st.info("Simulando escenario 'What-If'...")
                    
                    # Ejecutar y evaluar
                    simulations = what_if_simulation(df_original)
                    simulated_scores, simulated_f1 = evaluate_simulations(
                        simulations, df_original['Attrition'], model, le, scaler, cols_for_processing
                    )
                    
                    if simulated_scores:
                        st.success("🎉 Simulación What-If Completada.")
                        st.markdown(f"**Impacto: Accuracy con +10% Salario:** `{simulated_scores[0]:.4f}`")
                        st.markdown(f"**Impacto: F1-score con +10% Salario:** `{simulated_f1[0]:.4f}`")
                        
                        # Mostrar el impacto en la probabilidad promedio
                        sim_df_processed = preprocess_data(simulations[0].drop(columns=['Attrition'], errors='ignore'), cols_for_processing, le, scaler)
                        probabilidad_renuncia_sim = model.predict_proba(sim_df_processed)[:, 1]
                        
                        probabilidad_original = df_original['Probabilidad_Renuncia'].mean() if 'Probabilidad_Renuncia' in df_original.columns else 0
                        probabilidad_what_if = probabilidad_renuncia_sim.mean()
                        
                        st.metric(
                            label="Reducción Promedio de Probabilidad de Renuncia",
                            value=f"{(probabilidad_original - probabilidad_what_if):.4f}",
                            delta=f"{(probabilidad_original - probabilidad_what_if) * 100:.2f}%"
                        )
                        st.info("Un valor negativo en 'delta' indica un AUMENTO en la probabilidad, un valor positivo indica una REDUCCIÓN (deseable).")

# ============================
# Inicio de la Aplicación
# ============================
if __name__ == "__main__":
    main()


