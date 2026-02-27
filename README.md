#  Evaluación Predictiva de Precipitaciones Pluviales en México

Proyecto final de **Inteligencia de Negocios** — Primer Departamental.

##  Descripción

Sistema de predicción de precipitación diaria binarizada (>0 mm) a nivel nacional/regional, analizando el impacto de la radiación solar y contaminantes atmosféricos como posibles factores predictivos.

##  Estructura del Proyecto

```
├── proyecto_final_precipitaciones.py   # Script principal (completo)
├── dataset_nacional_clima.csv          # Clima diario base
├── meteorologica-radiacion.csv         # Datos meteorológicos diarios
├── data (1).csv                        # Datos mensuales CONAGUA
├── d3_aire01_49_1 (1).csv              # Inventario de emisiones (aire)
├── dataset_integrado.csv               # Dataset propio (generado)
├── mapa_estaciones.html                # Mapa interactivo Folium
├── correlaciones_region.png            # Correlaciones por región
├── matrices_confusion.png              # Matrices de confusión
├── f1_comparativo.png                  # Comparación F1-Score
├── feature_importance.png              # Importancia de variables
├── lstm_training_history.png           # Historial LSTM
└── README.md
```

##  Requisitos

```bash
pip install pandas scikit-learn xgboost tensorflow folium seaborn matplotlib
```

##  Ejecución

```bash
python proyecto_final_precipitaciones.py
```

El script genera automáticamente el dataset integrado, los gráficos y el mapa.

##  Datasets Utilizados

| # | Archivo | Fuente | Descripción |
|---|---------|--------|-------------|
| 1 | `dataset_nacional_clima.csv` | CONAGUA / Estaciones | Clima diario (Precipitación, Radiación, Temperaturas) |
| 2 | `d3_aire01_49_1 (1).csv` | SEMARNAT | Inventario Nacional de Emisiones (PM2.5, PM10, SO2, CO, NOx) |
| 3 | `data (1).csv` | CONAGUA | Datos climáticos mensuales por estado (1985-2025) |
| 4 | `meteorologica-radiacion.csv` | SMN | Promedios meteorológicos diarios nacionales |

##  Modelos Implementados

1. **Regresión Logística** — Modelo base lineal
2. **Random Forest** — Ensamble de árboles (300 estimadores)
3. **XGBoost** — Gradient Boosting optimizado
4. **LSTM** — Red neuronal recurrente (ventana temporal de 7 días)

##  Métricas Evaluadas

- Accuracy
- F1-Score
- AUC-ROC
- Matrices de Confusión

##  Visualizaciones

- Mapa interactivo Folium con marcadores por estado
- Heatmaps de correlación por región (Norte, Centro, Sur)
- Feature Importance (Random Forest + XGBoost)
- Barras comparativas de F1-Score
- Historial de entrenamiento LSTM

##  Autor

Leonardo Fiesco Ramirez de Arellano
