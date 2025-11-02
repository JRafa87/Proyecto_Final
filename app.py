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
        # df_reference_features ahora incluye todas las columnas EXCEPTO Attrition y se usará para el merge.
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
    # Nota: Aquí no se valida si faltan columnas que SOLO existen en el dataset de entrenamiento (e.g. EmployeeNumber si no se usa como feature). 
    # Solo se validan las columnas que el modelo XGBoost espera (model_columns).
    
    # Asegurarse de que las columnas que se usarán para el modelo estén presentes en el orden correcto
    
    # Rellenar nulos y codificar
    df_processed = df_processed.drop_duplicates()
    numeric_cols_for_fillna = df_processed.select_dtypes(include=np.number).columns.tolist()
    
    # Solo rellenamos las columnas numéricas que están en model_columns
    cols_to_fill = list(set(numeric_cols_for_fillna) & set(model_columns))
    df_processed[cols_to_fill] = df_processed[cols_to_fill].fillna(df_processed[cols_to_fill].mean())

    nominal_categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

    for col in nominal_categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).str.strip().str.upper()

            if col in categorical_mapping:
                # Mapeo de valores conocidos
                df_processed[col] = df_processed[col].map(categorical_mapping[col])

            # Rellenar valores desconocidos (si los hay) con el valor de 'DESCONOCIDO' o -1
            df_processed[col] = df_processed[col].fillna(categorical_mapping.get(col, {}).get('DESCONOCIDO', -1))
        
    # --- Ordenar y seleccionar solo las columnas de FEATURES ANTES DE ESCALAR ---
    df_to_scale = df_processed[model_columns].copy()
    
    # Escalado
    try:
        # El scaler espera solo las columnas de las features
        df_processed[model_columns] = scaler.transform(df_to_scale)
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}. ¿El scaler fue entrenado correctamente con la forma de la data?")
        return None

    # Finalmente, devolver el DataFrame solo con las columnas que el modelo espera y en el orden correcto
    return df_processed[model_columns]


# ============================
# 3. Simulaciones: Monte Carlo y What-If
# ============================
def monte_carlo_simulation(df_features, n_simulations=100, perturbation_range=(0.95, 1.05)):
    simulations = []
    # Usamos Age, MonthlyIncome, YearsAtCompany como variables sensibles a la simulación
    key_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany']
    
    for i in range(n_simulations):
        df_sim = df_features.copy()
        for col in key_cols:
            if col in df_sim.columns:
                # Asegurar que la perturbación se aplica solo a los valores en el DataFrame
                perturbation_factor = np.random.uniform(perturbation_range[0], perturbation_range[1], len(df_sim))
                df_sim[col] = df_sim[col] * perturbation_factor
        simulations.append(df_sim)
    return simulations


def what_if_simulation(df_features, perturbation_factor=1.10):
    df_sim = df_features.copy()
    if 'MonthlyIncome' in df_sim.columns:
        # Aumento del 10% en MonthlyIncome
        df_sim['MonthlyIncome'] *= perturbation_factor
    return [df_sim]


# ============================
# 4. Evaluación de Simulaciones
# ============================
def evaluate_simulations(simulated_datasets, true_labels_reference, model, categorical_mapping, scaler, model_feature_columns):
    scores = []
    f1_scores = []
    # Asegurar que true_labels_reference es un array o serie de ints (0 o 1)
    true_labels = true_labels_reference.values.astype(int) 

    for sim_data in simulated_datasets:
        # Preprocesar la data simulada usando el mismo pipeline de entrenamiento
        sim_data_processed = preprocess_data(sim_data, model_feature_columns, categorical_mapping, scaler)
        
        if sim_data_processed is None:
            # Si el preprocesamiento falla, retornamos vacíos
            st.warning("Preprocesamiento fallido en una simulación.")
            return [], []

        # Asegurar que los datos procesados tienen el mismo número de filas que las etiquetas reales
        if len(sim_data_processed) != len(true_labels):
            st.error("Error: El número de filas de los datos simulados no coincide con las etiquetas de referencia.")
            return [], []

        try:
            probabilidad_renuncia = model.predict_proba(sim_data_processed)[:, 1]
            predictions = (probabilidad_renuncia > 0.5).astype(int)
    
            acc = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, zero_division=0) # Manejo de división por cero
    
            scores.append(acc)
            f1_scores.append(f1)
        except Exception as e:
            st.error(f"Error al predecir o evaluar la simulación: {e}")
            return [], []

    return scores, f1_scores


# ============================
# 5. Exportar Resultados a Excel
# ============================
def export_results_to_excel(df, simulated_datasets=None, filename="simulation_results.xlsx"):
    # Verifica si existen columnas clave para el reporte. Si no, usa el índice.
    employee_key = 'EmployeeNumber' if 'EmployeeNumber' in df.columns else df.index.name or 'Index'
    
    with pd.ExcelWriter('temp.xlsx', engine='xlsxwriter') as output:
        # 1. Resultados Predicción
        df.to_excel(output, sheet_name='Resultados Predicción', index=False)

        # 2. Deserción Detallada
        df_desercion = df.copy()
        
        # Seleccionar las columnas relevantes (asegurando que existan)
        cols_for_report = [col for col in [employee_key, 'Prediction_Renuncia', 'Probabilidad_Renuncia', 'Department'] if col in df_desercion.columns]
        df_desercion = df_desercion[cols_for_report].copy()
        
        if 'Prediction_Renuncia' in df_desercion.columns:
            df_desercion['Deserción'] = df_desercion['Prediction_Renuncia'].apply(lambda x: 'Sí' if x == 1 else 'No')
            df_desercion = df_desercion.sort_values(by='Probabilidad_Renuncia', ascending=False)
            df_desercion.to_excel(output, sheet_name='Deserción Detallada', index=False)

        # 3. Simulaciones (si se tienen)
        if simulated_datasets:
            for i, sim_data in enumerate(simulated_datasets):
                # Incluir algunas columnas originales para referencia
                if employee_key in df.columns and employee_key not in sim_data.columns:
                    sim_data[employee_key] = df[employee_key]

                sim_data['Simulacion_ID'] = i + 1  
                sim_data.to_excel(output, sheet_name=f'Simulacion_{i+1}', index=False)

    # Leer el archivo para permitir la descarga
    try:
        with open('temp.xlsx', 'rb') as f:
            data = f.read()
        os.remove('temp.xlsx')
        return data
    except Exception as e:
        st.error(f"Error al generar el archivo Excel: {e}")
        return None


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


# ====================================
# 7. Funciones de Análisis Ejecutivo
# ====================================
def display_executive_analysis(df_results, title_suffix="Predicha"):
    st.markdown("---")
    st.subheader(f"Análisis de Deserción {title_suffix}")

    # --- 1. PORCENTAJE TOTAL DE DESERCIÓN (PREDICHO) ---
    total_attrition_count = df_results['Prediction_Renuncia'].sum()
    total_employees = len(df_results)
    total_attrition_percentage = (total_attrition_count / total_employees) * 100

    col_a, col_b = st.columns(2)
    col_a.metric(
        label=f"Porcentaje total de deserción ({title_suffix.lower()})", 
        value=f"{total_attrition_percentage:.2f}%", 
        delta=f"{total_attrition_count} empleados",
        delta_color="off"
    )
    col_b.empty() # Columna de relleno

    # --- 2. TOP 5 EMPLEADOS CON MAYOR RIESGO ---
    st.markdown("---")
    st.subheader("🚨 Top 5 Empleados con Mayor Riesgo de Deserción")

    # Intentar usar EmployeeNumber o Index como clave
    employee_key = 'EmployeeNumber' if 'EmployeeNumber' in df_results.columns else 'Index'

    # Manejar caso donde EmployeeNumber no es una columna
    df_temp = df_results.copy()
    if employee_key == 'Index':
        df_temp = df_temp.reset_index()
        df_temp = df_temp.rename(columns={df_temp.columns[0]: 'Index'})
        
    # 2.1 Obtener los 5 empleados con mayor probabilidad
    top_5_attrition = df_temp.sort_values(by='Probabilidad_Renuncia', ascending=False).head(5)

    # 2.2 Seleccionar solo las columnas clave para mostrar
    cols_to_display = [employee_key, 'Probabilidad_Renuncia', 'Department']
    
    top_5_display_cols = [col for col in cols_to_display if col in top_5_attrition.columns]
    top_5_display = top_5_attrition[top_5_display_cols].copy()

    # 2.3 Formatear la probabilidad para la tabla
    top_5_display['Probabilidad_Renuncia'] = top_5_display['Probabilidad_Renuncia'].map('{:.4f}'.format)
    
    # 2.4 Renombrar columnas para la visualización
    col_mapping = {
        employee_key: 'ID Empleado',
        'Probabilidad_Renuncia': 'Probabilidad de Renuncia',
        'Department': 'Departamento'
    }
    top_5_display = top_5_display.rename(columns=col_mapping)
    
    final_cols = ['ID Empleado', 'Probabilidad de Renuncia', 'Departamento']
    final_cols = [col for col in final_cols if col in top_5_display.columns]

    st.dataframe(top_5_display[final_cols], hide_index=True)
    
    st.markdown("---")
    
    # --- 3. DESERCIÓN POR ÁREA (DEPARTMENT) ---
    if 'Department' in df_results.columns:
        st.subheader("Deserción Predicha por Departamento (Tasa de Renuncia)")
        
        attrition_by_department = df_results.groupby('Department')['Prediction_Renuncia'].mean() * 100
        attrition_by_department = attrition_by_department.reset_index(name='Tasa_Desercion_Predicha')
        
        attrition_by_department['Tasa_Desercion_Predicha'] = attrition_by_department['Tasa_Desercion_Predicha'].map('{:.2f}%'.format)
        attrition_by_department.columns = ['Departamento', 'Tasa de Deserción Predicha']
        
        st.dataframe(attrition_by_department.sort_values(by='Departamento'), hide_index=True)
    else:
        st.warning(f"No se encontró la columna 'Department' en los datos de {title_suffix.lower()} para análisis por área.")


# ====================================
# 8. Funciones de Recomendaciones
# ====================================
def display_recommendations(df_results):
    st.markdown("---")
    st.header("💡 Recomendaciones Estratégicas")

    # Identificar el umbral de alto riesgo (ej. Probabilidad > 70%)
    high_risk_threshold = 0.70
    df_high_risk = df_results[df_results['Probabilidad_Renuncia'] > high_risk_threshold]
    
    if df_high_risk.empty:
        st.success("El riesgo de deserción es bajo. Mantenga las políticas actuales.")
        return

    # 1. Recomendación a Nivel Individual (Intervención)
    top_risk_count = min(10, len(df_high_risk))
    st.subheader("1. Intervención Individual Prioritaria")
    st.markdown(f"**Enfocarse en los {top_risk_count} empleados con la más alta probabilidad de renuncia** (Probabilidad > {high_risk_threshold * 100:.0f}%).")
    st.info("Acción sugerida: Realizar entrevistas de retención confidenciales para entender sus preocupaciones y ofrecer soluciones personalizadas (aumento salarial, flexibilidad, cambio de rol, o mentoría).")

    # 2. Recomendación a Nivel Departamental (Foco)
    if 'Department' in df_high_risk.columns:
        st.subheader("2. Foco Departamental")
        # Calcular el departamento con la mayor cantidad de empleados en alto riesgo
        risk_by_dept = df_high_risk.groupby('Department').size().sort_values(ascending=False)
        top_risk_dept = risk_by_dept.index[0]
        st.markdown(f"El departamento de **{top_risk_dept}** concentra el mayor número de empleados en alto riesgo ({risk_by_dept.iloc[0]} empleados).")
        st.warning(f"Acción sugerida: Evaluar la carga de trabajo, la gestión de líderes y los niveles de satisfacción general del equipo en **{top_risk_dept}**. Implementar programas de soporte específicos para ese departamento.")

    # 3. Recomendación a Nivel Global/Estratégico (Prevención)
    st.subheader("3. Estrategia Global de Prevención")
    if 'MonthlyIncome' in df_high_risk.columns:
        avg_high_risk_income = df_high_risk['MonthlyIncome'].mean()
        avg_total_income = df_results['MonthlyIncome'].mean()
        
        if avg_high_risk_income < avg_total_income * 0.9: # Si el ingreso promedio de riesgo es significativamente menor (ej. 10%)
            st.markdown("Se observa una correlación entre el bajo **MonthlyIncome** y el alto riesgo de deserción.")
            st.info("Acción sugerida: Revisar y ajustar la banda salarial para los roles de mayor riesgo, asegurando que el ingreso esté en línea o por encima del promedio del mercado para mitigar la insatisfacción salarial.")
        else:
            st.markdown("El riesgo no se concentra únicamente en bajos salarios.")
            st.info("Acción sugerida: Implementar encuestas de clima laboral anónimas enfocadas en **Work-Life Balance** y **Job Satisfaction** para identificar otros factores no salariales que impulsan la renuncia.")

# ============================
# 9. Interfaz Streamlit
# ============================
def main():
    st.set_page_config(page_title="Predicción y Simulación de Renuncia", layout="wide")
    st.title("📊 Modelo de Predicción y Simulación de Renuncia de Empleados")
    st.markdown("Carga tu archivo de datos para obtener predicciones. Las simulaciones usan una **data de referencia** cargada en el servidor para evaluación.")

    model, categorical_mapping, scaler, df_reference_features, true_labels_reference = load_model()
    if model is None:
        return 

    # Estas columnas DEBEN coincidir con las usadas en el entrenamiento y en el orden exacto.
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
        
        # Usamos df_features_uploaded para el preprocesamiento, pero df_original guarda las columnas originales
        processed_df = preprocess_data(df_features_uploaded, model_feature_columns, categorical_mapping, scaler)
        
        if processed_df is None:
            st.error("No se puede continuar con la predicción debido a un error de preprocesamiento.")
            return 

        st.header("1. Predicción con Datos Cargados")
        
        if st.button("🚀 Ejecutar Predicción y Evaluación"):
            st.info("Ejecutando el modelo sobre los datos cargados...")

            # --- EJECUTAR MODELO ---
            probabilidad_renuncia = model.predict_proba(processed_df)[:, 1]
            predictions = (probabilidad_renuncia > 0.5).astype(int)
            
            # Unir resultados a la data original
            df_original['Prediction_Renuncia'] = predictions
            df_original['Probabilidad_Renuncia'] = probabilidad_renuncia

            # --- 1. EVALUACIÓN Y MÉTRICAS BÁSICAS ---
            col1, col2 = st.columns(2)
            
            if 'Attrition' in df_original.columns:
                true_labels_uploaded = df_original['Attrition'].replace({'Yes': 1, 'No': 0}).astype(int)
                acc = accuracy_score(true_labels_uploaded, predictions)
                f1 = f1_score(true_labels_uploaded, predictions, zero_division=0)
                st.success("✅ Predicción y Evaluación de datos cargados Completadas!")
                col1.metric(label="Accuracy (Datos Cargados)", value=f"{acc:.4f}")
                col2.metric(label="F1-score (Datos Cargados)", value=f"{f1:.4f}")
            else:
                st.warning("⚠️ El archivo cargado no tiene la columna 'Attrition'. Solo se muestran las predicciones.")
            
            # --- 2. ANÁLISIS EJECUTIVO ---
            display_executive_analysis(df_original, title_suffix="Predicha")
            
            # --- 3. RECOMENDACIONES ---
            display_recommendations(df_original)

            # --- BOTÓN DE DESCARGA ---
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
                
                # --- ANÁLISIS EJECUTIVO DE MONTE CARLO (PRIMERA SIMULACIÓN) ---
                st.subheader("Análisis Detallado de Escenario (Simulación 1)")
                
                # 1. Preprocesar la data de la primera simulación
                sim_data_processed = preprocess_data(simulations[0], model_feature_columns, categorical_mapping, scaler)
                
                if sim_data_processed is not None:
                    # 2. Ejecutar la predicción
                    probabilidad_renuncia_sim = model.predict_proba(sim_data_processed)[:, 1]
                    predictions_sim = (probabilidad_renuncia_sim > 0.5).astype(int)
                    
                    # 3. Combinar resultados con las features originales (df_reference_features tiene ID Empleado y Department)
                    df_sim_results = df_reference_features.copy()
                    df_sim_results['Prediction_Renuncia'] = predictions_sim
                    df_sim_results['Probabilidad_Renuncia'] = probabilidad_renuncia_sim
                    
                    # 4. Mostrar análisis ejecutivo
                    display_executive_analysis(df_sim_results, title_suffix="Simulación MC")

                    # 5. RECOMENDACIONES
                    display_recommendations(df_sim_results)

                
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
                
                # --- ANÁLISIS EJECUTIVO DE WHAT-IF ---
                
                # 1. Preprocesar la data de la única simulación
                sim_data_processed = preprocess_data(simulations[0], model_feature_columns, categorical_mapping, scaler)
                
                if sim_data_processed is not None:
                    # 2. Ejecutar la predicción
                    probabilidad_renuncia_sim = model.predict_proba(sim_data_processed)[:, 1]
                    predictions_sim = (probabilidad_renuncia_sim > 0.5).astype(int)
                    
                    # 3. Combinar resultados con las features originales (df_reference_features tiene ID Empleado y Department)
                    df_sim_results = df_reference_features.copy()
                    df_sim_results['Prediction_Renuncia'] = predictions_sim
                    df_sim_results['Probabilidad_Renuncia'] = probabilidad_renuncia_sim
                    
                    # 4. Mostrar análisis ejecutivo
                    display_executive_analysis(df_sim_results, title_suffix="Escenario What-If (+10% Salario)")
                    
                    # 5. RECOMENDACIONES
                    display_recommendations(df_sim_results)

# ============================
# Inicio de la Aplicación
# ============================
if __name__ == "__main__":
    main()











