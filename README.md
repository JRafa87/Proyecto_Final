# Proyecto_Final

Este proyecto tiene como objetivo predecir la probabilidad de deserción de empleados, generar simulaciones de escenarios (Monte Carlo / What-If) y ofrecer recomendaciones estratégicas basadas en los resultados obtenidos.

La aplicación está desarrollada con Streamlit, utilizando modelos de Machine Learning (XGBoost) y herramientas de análisis de datos.

## 🏗️ Estructura del Proyecto

```text
.
├── app.py                     # Aplicación principal de Streamlit
├── models/                    # Modelos entrenados y artefactos del pipeline
│   ├── xgboost_model.pkl      # Modelo XGBoost entrenado
│   ├── scaler.pkl             # Scaler para normalizar las variables
│   ├── categorical_mapping.pkl # Mapeo de categorías para variables
│
├── data/                      # Datasets y datos de referencia
│   └── reference_data.csv     # Datos de referencia para simulaciones
│
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación del proyecto
``` 

## 🧰 TECNOLOGÍAS UTILIZADAS


**Lenguaje base:** Python 3.9+  
**Framework web:** Streamlit  
**Bibliotecas principales:**  
- Pandas / NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Joblib / Pickle  


## 🧮 Interacción con la Aplicación

📂 Carga de datos: Permite subir un archivo CSV o Excel con información de empleados.

🔮 Predicción: Calcula la probabilidad de renuncia para cada empleado según las variables cargadas.

🎲 Simulaciones: Ejecuta escenarios “What-If” o simulaciones Monte Carlo para evaluar estrategias.

📊 Resultados: Visualiza métricas, tablas y gráficos del riesgo de deserción por empleado y por área.

🧭 Recomendaciones: Muestra sugerencias automáticas sobre acciones de retención y prevención.


## 📈 Ejemplo de Uso

Carga el archivo data/reference_data.csv o un dataset propio con tus empleados.

Visualiza las probabilidades de deserción generadas por el modelo.

Ejecuta simulaciones para analizar el impacto de cambios en factores clave (por ejemplo: salario, satisfacción, horas extras).

Exporta los resultados y gráficos generados para análisis posterior.
