# Proyecto_Final

Este proyecto tiene como objetivo predecir la probabilidad de deserción de empleados, generar simulaciones y ofrecer recomendaciones basadas en los resultados obtenidos.

Estructura del Proyecto

La estructura de carpetas es la siguiente:

.
├── app.py                  # Archivo principal para ejecutar el pipeline con Streamlit
├── models/                 # Carpeta con los modelos entrenados y artefactos
│   ├── xgboost_model.pkl   # Modelo XGBoost entrenado
│   ├── scaler.pkl          # Scaler para normalizar las características
│   ├── categorical_mapping.pkl  # Mapeo de categorías para variables
├── data/                   # Carpeta con los datasets y datos de referencia
│   └── reference_data.csv  # Datos de referencia para simulaciones
├── requirements.txt        # Archivo de dependencias del proyecto
└── README.md               # Este archivo de documentación

Instrucciones de Uso

Instalar las dependencias:

El proyecto requiere las siguientes librerías de Python:

pandas
numpy
joblib
streamlit
sklearn
matplotlib

Ejecutar el pipeline:

Inicia la aplicación de Streamlit ejecutando:

streamlit run app.py

La interfaz de Streamlit te permitirá cargar un archivo CSV o Excel con los datos de los empleados y ejecutar las predicciones y simulaciones.

Interacción:

Puedes cargar datos externos para realizar predicciones.

Ejecutar simulaciones de Monte Carlo o What-If para evaluar escenarios alternativos.

Las recomendaciones sobre los empleados con mayor riesgo de deserción se mostrarán en la interfaz, así como análisis ejecutivos por área.
