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
def load_model():
    """
    Carga el modelo entrenado, el label encoder y el scaler.
    """
    model = joblib.load('models/xgboost_model.pkl')
    le = joblib.load('models/label_encoder.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, le, scaler


# ================================
# 2. Funciones de Preprocesamiento
# ================================
def preprocess_data(df, model_columns, le, scaler):
    """
    Preprocesa los datos: codificación y escalado.
    Si faltan columnas, muestra advertencia y detiene el flujo.
    """
    # Eliminar columnas que no están en la lista del modelo
    df = df[model_columns]

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Completar los valores nulos (imputar con la media de la columna)
    df = df.fillna(df.mean())  # Puedes cambiar esto por otro tipo de imputación si prefieres

    # Validar si faltan columnas
    missing_columns = set(model_columns) - set(df.columns)
    if missing_columns:
        st.warning(f"Faltan las siguientes columnas: {missing_columns}")
        return None  # Detener ejecución si faltan columnas
    
    # Codificación de variables categóricas
    for col in ['Gender', 'BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'OverTime']:
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))  # Codificación
    
    # Escalado de las variables numéricas
    numeric_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'JobLevel', 'MonthlyIncome', 'MonthlyRate', 
                       'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
                       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
                       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    df[numeric_columns] = scaler.transform(df[numeric_columns])  # Escalado
    
    return df


# ============================
# 3. Simulaciones: Monte Carlo y What-If
# ============================
def monte_carlo_simulation(df, n_simulations=100, perturbation_range=(0.95, 1.05)):
    """
    Realiza simulaciones de Monte Carlo generando perturbaciones aleatorias sobre las variables clave.
    """
    simulations = []
    for i in range(n_simulations):
        df_sim = df.copy()
        for col in ['Age', 'MonthlyIncome', 'YearsAtCompany']:
            if col in df_sim.columns:
                perturbation_factor = np.random.uniform(perturbation_range[0], perturbation_range[1], len(df_sim))
                df_sim[col] = df_sim[col] * perturbation_factor
        simulations.append(df_sim)
    return simulations


def what_if_simulation(df, perturbation_factor=1.10):
    """
    Simula escenarios 'What-If' variando un parámetro clave (por ejemplo, aumentar el salario en un 10%).
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
    
    for sim_data in simulated_datasets:
        sim_data_processed = preprocess_data(sim_data, model_columns, le, scaler)
        if sim_data_processed is None:
            return [], []  # Detener ejecución si preprocesamiento falla
        
        probabilidad_renuncia = model.predict_proba(sim_data_processed)[:, 1]
        predictions = (probabilidad_renuncia > 0.5).astype(int)  # Umbral de 0.5 para clasificación binaria
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        scores.append(acc)
        f1_scores.append(f1)
    
    return scores, f1_scores


# ============================
# 5. Exportar Resultados a Excel
# ============================
def export_results_to_excel(df, simulated_scores, simulated_f1, filename="simulation_results.xlsx"):
    """
    Exporta los resultados de las predicciones y simulaciones a un archivo Excel.
    """
    output_path = os.path.join('data', 'simulations', filename)
    df['Simulated_Accuracy'] = simulated_scores
    df['Simulated_F1'] = simulated_f1
    df.to_excel(output_path, index=False)
    return output_path


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
    ax[0].set_title('Distribución de Accuracy')
    ax[0].set_xlabel('Accuracy')
    ax[0].set_ylabel('Frecuencia')

    # Graficar F1-score
    ax[1].hist(simulated_f1, bins=10, color='lightcoral', edgecolor='black')
    ax[1].set_title('Distribución de F1-score')
    ax[1].set_xlabel('F1-score')
    ax[1].set_ylabel('Frecuencia')

    plt.tight_layout()
    st.pyplot(fig)


# ============================
# 7. Interfaz de Streamlit
# ============================
def main():
    # Título de la aplicación
    st.title("Modelo de Predicción de Renuncia de Empleados")

    # Cargar el modelo, el encoder y el scaler
    model, le, scaler = load_model()

    # Lista de columnas que el modelo espera
    model_columns = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
        'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager', 'IntencionPermanencia',
        'CargaLaboralPercibida', 'SatisfaccionSalarial', 'ConfianzaEmpresa', 'NumeroTardanzas', 'NumeroFaltas'
    ]

    # Opción para cargar el archivo
    uploaded_file = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)  # Cargar el archivo como DataFrame
        st.write("Datos cargados:", df)

        # Verificación de columnas faltantes antes de ejecutar cualquier simulación
        processed_df = preprocess_data(df, model_columns, le, scaler)
        if processed_df is None:
            return  # Detener ejecución si faltan columnas

        # Mostrar opción para predecir con los datos cargados
        if st.button("Predecir con Datos Cargados"):  # Botón para ejecutar el modelo con los datos cargados
            st.write("Ejecutando el modelo sobre los datos cargados...")
            probabilidad_renuncia = model.predict_proba(processed_df)[:, 1]
            predictions = (probabilidad_renuncia > 0.5).astype(int)  # Umbral de 0.5 para clasificación binaria
            st.write("Predicciones de Renuncia: ", predictions)

            # Evaluar el modelo
            acc = accuracy_score(df['Attrition'], predictions)
            f1 = f1_score(df['Attrition'], predictions)
            st.write(f"Accuracy: {acc}")
            st.write(f"F1-score: {f1}")

            # Opción para descargar los resultados
            output_file = export_results_to_excel(df, [acc]*len(df), [f1]*len(df))
            st.download_button("Descargar Resultados de Predicción", data=open(output_file, "rb").read(), file_name="predicciones_resultados.xlsx")

        # Mostrar opción para ejecutar Monte Carlo o What-If
        simulation_option = st.radio("Selecciona tipo de simulación:", ["Monte Carlo", "What-If"])

        if simulation_option == "Monte Carlo":
            if st.button("Ejecutar Monte Carlo"):  # Botón para ejecutar Monte Carlo
                st.write("Simulando Monte Carlo...")
                simulations = monte_carlo_simulation(df)
                simulated_scores, simulated_f1 = evaluate_simulations(simulations, df['Attrition'], model, le, scaler, model_columns)
                st.write("Resultados de Simulaciones - Accuracy:", simulated_scores)
                st.write("Resultados de Simulaciones - F1-score:", simulated_f1)
                plot_metrics(simulated_scores, simulated_f1)
                output_file = export_results_to_excel(df, simulated_scores, simulated_f1)
                st.download_button("Descargar Resultados de Monte Carlo", data=open(output_file, "rb").read(), file_name="monte_carlo_results.xlsx")

        elif simulation_option == "What-If":
            if st.button("Ejecutar What-If"):  # Botón para ejecutar What-If
                st.write("Simulando What-If...")
                simulations = what_if_simulation(df)
                simulated_scores, simulated_f1 = evaluate_simulations(simulations, df['Attrition'], model, le, scaler, model_columns)
                st.write("Resultados de Simulaciones - Accuracy:", simulated_scores)
                st.write("Resultados de Simulaciones - F1-score:", simulated_f1)
                plot_metrics(simulated_scores, simulated_f1)
                output_file = export_results_to_excel(df, simulated_scores, simulated_f1)
                st.download_button("Descargar Resultados de What-If", data=open(output_file, "rb").read(), file_name="what_if_results.xlsx")


if __name__ == "__main__":
    main()

