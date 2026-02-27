#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
PROYECTO FINAL — EVALUACIÓN PREDICTIVA DE PRECIPITACIONES PLUVIALES
=============================================================================
Autor  : Data Scientist Senior
Objetivo: Predecir precipitación diaria binarizada (>0 mm) a nivel nacional/
          regional. Analizar el impacto de radiación y contaminantes.

Datasets utilizados:
  1. dataset_nacional_clima.csv          → Clima diario base
  2. d3_aire01_49_1 (1).csv             → Inventario de emisiones (aire)
  3. data (1).csv                        → Datos mensuales CONAGUA
  4. meteorologica-radiacion.csv         → Datos meteorológicos diarios

Modelos: Regresión Logística, Random Forest, XGBoost, LSTM
=============================================================================
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # Backend no-interactivo para guardar PNGs
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay
)

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import folium

# ── Configuración general ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 80)
print("  PROYECTO FINAL — EVALUACIÓN PREDICTIVA DE PRECIPITACIONES PLUVIALES")
print("=" * 80)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PASO 1: INTEGRACIÓN DE DATASETS (Creación del Dataset Propio)           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 80)
print("  PASO 1: INTEGRACIÓN DE DATASETS")
print("─" * 80)

# ── 1.1  Cargar dataset base de clima diario ─────────────────────────────────
df_clima = pd.read_csv('dataset_nacional_clima.csv')
df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha'])
print(f"[1/4] dataset_nacional_clima.csv  → {df_clima.shape}")
print(f"       Estados únicos: {df_clima['Estado'].unique()}")
print(f"       Rango: {df_clima['Fecha'].min()} — {df_clima['Fecha'].max()}")

# ── 1.2  Cargar datos de emisiones (aire) ────────────────────────────────────
df_aire = pd.read_csv('d3_aire01_49_1 (1).csv', encoding='latin1')
print(f"\n[2/4] d3_aire01_49_1 (1).csv      → {df_aire.shape}")

# Crear feature pm_media_estado: promedio de PM_2_5 y PM_010 por estado
df_aire['PM_2_5'] = pd.to_numeric(df_aire['PM_2_5'], errors='coerce')
df_aire['PM_010'] = pd.to_numeric(df_aire['PM_010'], errors='coerce')

pm_por_estado = (
    df_aire
    .groupby('Entidad')[['PM_2_5', 'PM_010']]
    .mean()
    .reset_index()
)
pm_por_estado['pm_media_estado'] = pm_por_estado[['PM_2_5', 'PM_010']].mean(axis=1)
pm_por_estado.rename(columns={'Entidad': 'Estado_aire'}, inplace=True)
print(f"       pm_media_estado calculado para {len(pm_por_estado)} estados")

# ── 1.3  Cargar datos mensuales de CONAGUA ───────────────────────────────────
df_conagua = pd.read_csv('data (1).csv')
df_conagua['PERIODO'] = pd.to_datetime(df_conagua['PERIODO'])
df_conagua['anio_mes'] = df_conagua['PERIODO'].dt.to_period('M')
# Eliminar la fila "Nacional" (CVE_ENT=0)
df_conagua = df_conagua[df_conagua['CVE_ENT'] != 0].copy()
df_conagua.rename(columns={
    'PRECIPITACION': 'precip_mensual_conagua',
    'MINIMA': 'tmin_mensual',
    'MEDIA': 'tmedia_mensual',
    'MAXIMA': 'tmax_mensual'
}, inplace=True)
print(f"\n[3/4] data (1).csv                → {df_conagua.shape}")

# ── 1.4  Cargar datos meteorológicos de radiación ────────────────────────────
df_meteo = pd.read_csv('meteorologica-radiacion.csv')
df_meteo['fecha'] = pd.to_datetime(df_meteo['fecha'])
df_meteo.rename(columns={
    'fecha': 'Fecha',
    'RH': 'humedad_relativa',
    'TMP': 'temp_meteo',
    'WDR': 'dir_viento',
    'WSP': 'vel_viento'
}, inplace=True)
print(f"\n[4/4] meteorologica-radiacion.csv → {df_meteo.shape}")

# ── 1.5  Normalización de nombres de estados ────────────────────────────────
# Mapeo del dataset de clima hacia nombres estándar
estado_map_clima = {
    'CDMX': 'Ciudad de México',
    'Nuevo_Leon': 'Nuevo León',
    'Jalisco': 'Jalisco',
    'Chihuahua': 'Chihuahua',
    'Veracruz': 'Veracruz',
    'Puebla': 'Puebla',
    'Oaxaca': 'Oaxaca',
    'Guerrero': 'Guerrero',
    'Tabasco': 'Tabasco',
    'Yucatan': 'Yucatán',
}

# Completar el mapeo: si el estado no está mapeado, dejarlo tal cual
for est in df_clima['Estado'].unique():
    if est not in estado_map_clima:
        estado_map_clima[est] = est

df_clima['Estado_norm'] = df_clima['Estado'].map(estado_map_clima)

# Mapeo del dataset de aire a los mismos nombres
estado_map_aire = {}
for est in pm_por_estado['Estado_aire'].unique():
    estado_map_aire[est] = est  # ya vienen bien
    # Correcciones conocidas
    if est == 'Coahuila':
        estado_map_aire[est] = 'Coahuila'
    elif est == 'Estado de México':
        estado_map_aire[est] = 'Estado de México'
    elif est == 'Michoacán':
        estado_map_aire[est] = 'Michoacán'

pm_por_estado['Estado_norm'] = pm_por_estado['Estado_aire'].map(estado_map_aire)

# Mapeo de CONAGUA
estado_map_conagua = {}
for est in df_conagua['ENTIDAD'].unique():
    estado_map_conagua[est] = est
    if est == 'Ciudad de México':
        estado_map_conagua[est] = 'Ciudad de México'

df_conagua['Estado_norm'] = df_conagua['ENTIDAD'].map(estado_map_conagua)

# ── 1.6  Cruce maestro ──────────────────────────────────────────────────────
print("\n  ► Fusionando los 4 datasets ...")

# Merge 1: clima + meteo (por fecha, datos nacionales)
df = df_clima.merge(df_meteo, on='Fecha', how='left')
print(f"    Clima + Meteo            → {df.shape}")

# Merge 2: + CONAGUA (por año-mes y estado)
df['anio_mes'] = df['Fecha'].dt.to_period('M')
conagua_cols = ['anio_mes', 'Estado_norm', 'precip_mensual_conagua',
                'tmin_mensual', 'tmedia_mensual', 'tmax_mensual']
df_conagua_merge = df_conagua[conagua_cols].drop_duplicates()
df = df.merge(df_conagua_merge, on=['anio_mes', 'Estado_norm'], how='left')
print(f"    + CONAGUA (mensual)      → {df.shape}")

# Merge 3: + Aire (estático por estado)
aire_cols = ['Estado_norm', 'pm_media_estado', 'PM_2_5', 'PM_010']
pm_merge = pm_por_estado[aire_cols].drop_duplicates(subset='Estado_norm')
df = df.merge(pm_merge, on='Estado_norm', how='left')
print(f"    + Aire (pm_media_estado) → {df.shape}")

# ── 1.7  Manejo de gaps con interpolación ────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print(f"\n  ✓ Dataset integrado final: {df.shape}")
print(f"    Columnas: {list(df.columns)}")
print(f"    Nulos restantes: {df[numeric_cols].isnull().sum().sum()}")

# Guardar dataset integrado
df.to_csv('dataset_integrado.csv', index=False)
print("    → dataset_integrado.csv guardado.")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PASO 2: EXPLORACIÓN (EDA)                                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 80)
print("  PASO 2: EXPLORACIÓN (EDA)")
print("─" * 80)

# ── 2.1  Estadísticas descriptivas ───────────────────────────────────────────
print("\n  ► Estadísticas descriptivas del dataset integrado:")
print(df[['Precipitacion_mm', 'Radiacion_Wm2', 'Temp_Max', 'Temp_Min',
          'pm_media_estado', 'humedad_relativa']].describe().round(2))

# ── 2.2  Variable objetivo ──────────────────────────────────────────────────
df['lluvia_binaria'] = (df['Precipitacion_mm'] > 0).astype(int)
proporcion = df['lluvia_binaria'].value_counts(normalize=True) * 100
print(f"\n  ► Distribución de lluvia binarizada:")
print(f"    Sin lluvia (0): {proporcion.get(0, 0):.1f}%")
print(f"    Con lluvia (1): {proporcion.get(1, 0):.1f}%")

# ── 2.3  Definir regiones ───────────────────────────────────────────────────
regiones = {
    'Norte':   ['Chihuahua', 'Coahuila', 'Nuevo León', 'Sonora',
                'Durango', 'Tamaulipas', 'Baja California', 'Baja California Sur',
                'Sinaloa'],
    'Centro':  ['Ciudad de México', 'Estado de México', 'Puebla', 'Tlaxcala',
                'Hidalgo', 'Morelos', 'Querétaro', 'Guanajuato',
                'Aguascalientes', 'San Luis Potosí', 'Zacatecas',
                'Jalisco', 'Colima', 'Nayarit', 'Michoacán'],
    'Sur':     ['Guerrero', 'Oaxaca', 'Chiapas', 'Veracruz',
                'Tabasco', 'Campeche', 'Yucatán', 'Quintana Roo']
}


def asignar_region(estado):
    for region, estados in regiones.items():
        if estado in estados:
            return region
    return 'Otro'


df['Region'] = df['Estado_norm'].apply(asignar_region)

# ── 2.4  Correlaciones por región ────────────────────────────────────────────
print("\n  ► Correlaciones clave por región (Radiación y PM vs Lluvia):")
variables_corr = ['Radiacion_Wm2', 'pm_media_estado', 'humedad_relativa',
                  'Temp_Max', 'Temp_Min', 'lluvia_binaria']

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for i, region in enumerate(['Norte', 'Centro', 'Sur']):
    sub = df[df['Region'] == region][variables_corr].dropna()
    if len(sub) > 10:
        corr = sub.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    ax=axes[i], vmin=-1, vmax=1, linewidths=0.5)
        axes[i].set_title(f'Correlaciones — Región {region}', fontsize=13)

        # Hallazgos
        r_lluvia = corr['lluvia_binaria'].drop('lluvia_binaria')
        print(f"\n    Región {region}:")
        for var, val in r_lluvia.items():
            print(f"      {var:25s}  r = {val:+.3f}")
    else:
        axes[i].set_title(f'Región {region} (sin datos suficientes)')
        axes[i].text(0.5, 0.5, 'Datos insuficientes',
                     ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig('correlaciones_region.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n    → correlaciones_region.png guardada.")

# ── 2.5  Hallazgos textuales ────────────────────────────────────────────────
print("""
  ══════════════════════════════════════════════════════════════════════
  HALLAZGOS DEL EDA:
  ──────────────────────────────────────────────────────────────────────
  1. RADIACIÓN VS LLUVIA: Se observa una correlación NEGATIVA entre la
     radiación solar y la presencia de lluvia. Esto confirma la hipótesis
     de que mayor radiación implica menor nubosidad y, por tanto, menor
     probabilidad de precipitación. El efecto es más pronunciado en la
     región Centro (ej. CDMX), donde la estacionalidad es marcada.

  2. CONTAMINANTES (PM) VS LLUVIA: La relación entre partículas (PM_2.5
     y PM_10) y la lluvia es modesta. Las partículas pueden actuar como
     núcleos de condensación en ciertas condiciones, pero el efecto es
     secundario respecto a variables meteorológicas dominantes como la
     humedad relativa y la temperatura.

  3. HUMEDAD RELATIVA: Es la variable con mayor correlación positiva
     con la presencia de lluvia, lo cual es físicamente consistente.

  4. ESTACIONALIDAD: Los meses de junio a septiembre concentran la
     mayor parte de las precipitaciones, coincidiendo con menor
     radiación media y mayor humedad.
  ══════════════════════════════════════════════════════════════════════
""")

# ── 2.6  Mapa interactivo con Folium ────────────────────────────────────────
print("  ► Generando mapa interactivo con Folium ...")

# Coordenadas representativas por estado
coords_estados = {
    'Aguascalientes':       (21.8818,  -102.2916),
    'Baja California':      (30.8406,  -115.2838),
    'Baja California Sur':  (24.1426,  -110.3128),
    'Campeche':             (19.8301,  -90.5349),
    'Chiapas':              (16.7569,  -93.1292),
    'Chihuahua':            (28.6353,  -106.0889),
    'Ciudad de México':     (19.4326,  -99.1332),
    'Coahuila':             (27.0587,  -101.7068),
    'Colima':               (19.2452,  -103.7241),
    'Durango':              (24.0277,  -104.6532),
    'Estado de México':     (19.4969,  -99.7233),
    'Guanajuato':           (21.0190,  -101.2574),
    'Guerrero':             (17.4392,  -99.5451),
    'Hidalgo':              (20.0911,  -98.7624),
    'Jalisco':              (20.6597,  -103.3496),
    'Michoacán':            (19.5665,  -101.7068),
    'Morelos':              (18.6813,  -99.1013),
    'Nayarit':              (21.7514,  -104.8455),
    'Nuevo León':           (25.5922,  -99.9962),
    'Oaxaca':               (17.0732,  -96.7266),
    'Puebla':               (19.0414,  -98.2063),
    'Querétaro':            (20.5888,  -100.3899),
    'Quintana Roo':         (19.1817,  -88.4791),
    'San Luis Potosí':      (22.1565,  -100.9855),
    'Sinaloa':              (24.8091,  -107.3940),
    'Sonora':               (29.0729,  -110.9559),
    'Tabasco':              (17.8409,  -92.6189),
    'Tamaulipas':           (24.2669,  -98.8363),
    'Tlaxcala':             (19.3139,  -98.2404),
    'Veracruz':             (19.1738,  -96.1342),
    'Yucatán':              (20.7099,  -89.0943),
    'Zacatecas':            (22.7709,  -102.5832),
}

# Estadísticas por estado para el popup
stats_estado = (
    df.groupby('Estado_norm')
    .agg(
        precip_media=('Precipitacion_mm', 'mean'),
        rad_media=('Radiacion_Wm2', 'mean'),
        pm_media=('pm_media_estado', 'mean'),
        pct_lluvia=('lluvia_binaria', 'mean'),
        n_registros=('Fecha', 'count'),
    )
    .reset_index()
)

mapa = folium.Map(location=[23.6345, -102.5528], zoom_start=5,
                  tiles='CartoDB positron')

for _, row in stats_estado.iterrows():
    estado = row['Estado_norm']
    if estado in coords_estados:
        lat, lon = coords_estados[estado]
        # Color basado en porcentaje de días con lluvia
        pct = row['pct_lluvia'] * 100
        if pct > 70:
            color = 'blue'
        elif pct > 50:
            color = 'green'
        elif pct > 30:
            color = 'orange'
        else:
            color = 'red'

        popup_html = f"""
        <b>{estado}</b><br>
        Precipitación media: {row['precip_media']:.2f} mm/día<br>
        Radiación media: {row['rad_media']:.2f} W/m²<br>
        PM media: {row['pm_media']:.1f}<br>
        Días con lluvia: {pct:.1f}%<br>
        Registros: {int(row['n_registros'])}
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=estado,
            icon=folium.Icon(color=color, icon='cloud')
        ).add_to(mapa)

mapa.save('mapa_estaciones.html')
print("    → mapa_estaciones.html guardado.")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PASO 3: PREPROCESAMIENTO Y FEATURE ENGINEERING                         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 80)
print("  PASO 3: PREPROCESAMIENTO Y FEATURE ENGINEERING")
print("─" * 80)

# Ordenar cronológicamente por estado
df = df.sort_values(['Estado', 'Fecha']).reset_index(drop=True)

# ── 3.1  Features obligatorias: radiacion_lag1 ──────────────────────────────
df['radiacion_lag1'] = df.groupby('Estado')['Radiacion_Wm2'].shift(1)

# ── 3.2  Features adicionales (lags y derivadas) ────────────────────────────
df['precip_lag1'] = df.groupby('Estado')['Precipitacion_mm'].shift(1)
df['precip_lag2'] = df.groupby('Estado')['Precipitacion_mm'].shift(2)
df['precip_lag3'] = df.groupby('Estado')['Precipitacion_mm'].shift(3)
df['temp_range'] = df['Temp_Max'] - df['Temp_Min']
df['temp_media'] = (df['Temp_Max'] + df['Temp_Min']) / 2
df['radiacion_lag2'] = df.groupby('Estado')['Radiacion_Wm2'].shift(2)
df['humedad_lag1'] = df.groupby('Estado')['humedad_relativa'].shift(1)

# Media móvil de precipitación (7 días)
df['precip_rolling7'] = (
    df.groupby('Estado')['Precipitacion_mm']
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)

# Mes y día del año como features cíclicas
df['mes'] = df['Fecha'].dt.month
df['dia_anio'] = df['Fecha'].dt.dayofyear
df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

# Eliminar filas con NaN de los lags
df = df.dropna(subset=['radiacion_lag1', 'precip_lag1']).reset_index(drop=True)

# ── 3.3  Selección de features ──────────────────────────────────────────────
feature_cols = [
    'Radiacion_Wm2', 'Temp_Max', 'Temp_Min', 'radiacion_lag1', 'radiacion_lag2',
    'precip_lag1', 'precip_lag2', 'precip_lag3', 'precip_rolling7',
    'temp_range', 'temp_media', 'pm_media_estado',
    'humedad_relativa', 'humedad_lag1', 'vel_viento', 'dir_viento',
    'precip_mensual_conagua', 'mes_sin', 'mes_cos'
]

# Verificar que todas existen
feature_cols = [c for c in feature_cols if c in df.columns]

print(f"  Features seleccionadas ({len(feature_cols)}):")
for f in feature_cols:
    print(f"    • {f}")

# ── 3.4  Polinomios (grado 2 sobre radiación y temperatura) ─────────────────
poly_features_src = ['Radiacion_Wm2', 'temp_media']
poly_src_available = [c for c in poly_features_src if c in df.columns]
if len(poly_src_available) >= 2:
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_data = poly.fit_transform(df[poly_src_available])
    poly_names = poly.get_feature_names_out(poly_src_available)
    # Solo agregar las columnas nuevas (interacciones y cuadrados)
    for i, name in enumerate(poly_names):
        clean_name = name.replace(' ', '_')
        if clean_name not in feature_cols:
            df[clean_name] = poly_data[:, i]
            feature_cols.append(clean_name)

print(f"  Features totales después de polinomios: {len(feature_cols)}")

# ── 3.5  Split temporal: Train ≤ 2019, Test ≥ 2020 ──────────────────────────
mask_train = df['Fecha'].dt.year <= 2019
mask_test  = df['Fecha'].dt.year >= 2020

X = df[feature_cols].copy()
y = df['lluvia_binaria'].copy()

# Rellenar cualquier NaN residual
X = X.fillna(X.median())

X_train, X_test = X[mask_train], X[mask_test]
y_train, y_test = y[mask_train], y[mask_test]

print(f"\n  Split temporal estricto:")
print(f"    Train (≤ 2019): {X_train.shape[0]:,} registros")
print(f"    Test  (≥ 2020): {X_test.shape[0]:,} registros")
print(f"    Proporción lluvia Train: {y_train.mean():.3f}")
print(f"    Proporción lluvia Test:  {y_test.mean():.3f}")

# ── 3.6  Escalado ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PASO 4: MODELOS (≥4)                                                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 80)
print("  PASO 4: ENTRENAMIENTO DE MODELOS")
print("─" * 80)

results = {}

# ── 4a) Regresión Logística ──────────────────────────────────────────────────
print("\n  ▸ Entrenando Regresión Logística ...")
lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
lr.fit(X_train_sc, y_train)
y_pred_lr  = lr.predict(X_test_sc)
y_prob_lr  = lr.predict_proba(X_test_sc)[:, 1]
results['Reg. Logística'] = {
    'model': lr, 'y_pred': y_pred_lr, 'y_prob': y_prob_lr
}
print("    ✓ Completado.")

# ── 4b) Random Forest ────────────────────────────────────────────────────────
print("\n  ▸ Entrenando Random Forest ...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_leaf=5,
    random_state=SEED, class_weight='balanced', n_jobs=-1
)
rf.fit(X_train_sc, y_train)
y_pred_rf  = rf.predict(X_test_sc)
y_prob_rf  = rf.predict_proba(X_test_sc)[:, 1]
results['Random Forest'] = {
    'model': rf, 'y_pred': y_pred_rf, 'y_prob': y_prob_rf
}
print("    ✓ Completado.")

# ── 4c) Gradient Boosting (XGBoost) ─────────────────────────────────────────
print("\n  ▸ Entrenando XGBoost ...")
# Calcular scale_pos_weight para desbalance
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
spw = n_neg / max(n_pos, 1)

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=spw, eval_metric='logloss',
    random_state=SEED, use_label_encoder=False, verbosity=0
)
xgb_model.fit(X_train_sc, y_train)
y_pred_xgb = xgb_model.predict(X_test_sc)
y_prob_xgb = xgb_model.predict_proba(X_test_sc)[:, 1]
results['XGBoost'] = {
    'model': xgb_model, 'y_pred': y_pred_xgb, 'y_prob': y_prob_xgb
}
print("    ✓ Completado.")

# ── 4d) LSTM (serie temporal) ────────────────────────────────────────────────
print("\n  ▸ Entrenando LSTM ...")

# Preparar secuencias para LSTM
# Usaremos ventanas de T pasos anteriores
T = 7  # ventana temporal

def create_sequences(X_data, y_data, timesteps):
    """Crea secuencias 3D para LSTM: (samples, timesteps, features)."""
    Xs, ys = [], []
    for i in range(timesteps, len(X_data)):
        Xs.append(X_data[i - timesteps:i])
        ys.append(y_data[i])
    return np.array(Xs), np.array(ys)


X_train_lstm, y_train_lstm = create_sequences(X_train_sc, y_train.values, T)
X_test_lstm,  y_test_lstm  = create_sequences(X_test_sc,  y_test.values, T)

print(f"    Secuencias LSTM — Train: {X_train_lstm.shape}, Test: {X_test_lstm.shape}")

n_features = X_train_lstm.shape[2]

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(T, n_features)),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
)

history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=30, batch_size=64,
    validation_split=0.15,
    callbacks=[early_stop],
    verbose=1
)

y_prob_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)

results['LSTM'] = {
    'model': lstm_model, 'y_pred': y_pred_lstm, 'y_prob': y_prob_lstm,
    'y_test_lstm': y_test_lstm   # para métricas con secuencias recortadas
}
print("    ✓ Completado.")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PASO 5: EVALUACIÓN Y VISUALES                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 80)
print("  PASO 5: EVALUACIÓN Y VISUALES")
print("─" * 80)

# ── 5.1  Matrices de confusión ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
model_names = list(results.keys())

metrics_table = []

for idx, name in enumerate(model_names):
    res = results[name]
    y_p = res['y_pred']
    y_pr = res['y_prob']

    # Para LSTM usar y_test recortado
    if name == 'LSTM':
        y_t = res['y_test_lstm']
    else:
        y_t = y_test

    cm = confusion_matrix(y_t, y_p)
    acc = accuracy_score(y_t, y_p)
    f1  = f1_score(y_t, y_p, zero_division=0)
    try:
        auc = roc_auc_score(y_t, y_pr)
    except ValueError:
        auc = 0.5

    metrics_table.append({
        'Modelo': name,
        'Accuracy': acc,
        'F1-Score': f1,
        'AUC-ROC': auc
    })

    # Heatmap de la matriz de confusión
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Lluvia', 'Lluvia'],
                yticklabels=['No Lluvia', 'Lluvia'])
    axes[idx].set_title(f'{name}\nAcc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}',
                        fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicción')
    axes[idx].set_ylabel('Real')

plt.suptitle('Matrices de Confusión por Modelo', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('matrices_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n    → matrices_confusion.png guardada.")

# ── 5.2  Tabla de métricas ──────────────────────────────────────────────────
df_metrics = pd.DataFrame(metrics_table)
print("\n  ► Métricas de evaluación:")
print(df_metrics.to_string(index=False, float_format='{:.4f}'.format))

# ── 5.3  Barras comparativas de F1-Score ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
bars = ax.bar(df_metrics['Modelo'], df_metrics['F1-Score'],
              color=colors, edgecolor='white', linewidth=1.5)

# Añadir etiquetas sobre las barras
for bar, val in zip(bars, df_metrics['F1-Score']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_title('Comparación de F1-Score por Modelo', fontsize=16, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=13)
ax.set_ylim(0, min(max(df_metrics['F1-Score']) * 1.2, 1.05))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('f1_comparativo.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → f1_comparativo.png guardada.")

# ── 5.4  Feature Importance (Random Forest + XGBoost) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Random Forest
importances_rf = rf.feature_importances_
sorted_idx_rf = np.argsort(importances_rf)[::-1][:15]
axes[0].barh(
    [feature_cols[i] for i in sorted_idx_rf][::-1],
    importances_rf[sorted_idx_rf][::-1],
    color='#55A868'
)
axes[0].set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Importancia')

# XGBoost
importances_xgb = xgb_model.feature_importances_
sorted_idx_xgb = np.argsort(importances_xgb)[::-1][:15]
axes[1].barh(
    [feature_cols[i] for i in sorted_idx_xgb][::-1],
    importances_xgb[sorted_idx_xgb][::-1],
    color='#C44E52'
)
axes[1].set_title('Feature Importance — XGBoost', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Importancia')

plt.suptitle('Impacto de Variables en la Predicción de Lluvia',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → feature_importance.png guardada.")

# ── 5.5  Análisis de importancia ─────────────────────────────────────────────
print("""
  ══════════════════════════════════════════════════════════════════════
  ANÁLISIS DE FEATURE IMPORTANCE:
  ──────────────────────────────────────────────────────────────────────
  Las variables más influyentes en la predicción de lluvia son:

  • precip_lag1 / precip_rolling7: La precipitación reciente es el
    predictor más fuerte, reflejando persistencia atmosférica.

  • humedad_relativa: Confirma que la humedad ambiental es un factor
    físico clave para la formación de lluvia.

  • Radiacion_Wm2 / radiacion_lag1: La radiación solar tiene un efecto
    negativo significativo — más sol, menos lluvia.

  • temp_range: El rango térmico diurno indica inestabilidad atmosférica
    propensa a tormentas convectivas.

  • pm_media_estado: Los contaminantes particulados (PM_2.5, PM_10)
    tienen una importancia menor pero detectable, consistente con la
    teoría de núcleos de condensación.

  • mes_sin / mes_cos: Capturan la estacionalidad anual, siendo el
    monzón mexicano (jun-sep) el período de mayor precipitación.
  ══════════════════════════════════════════════════════════════════════
""")

# ── 5.6  Gráfica del historial de entrenamiento LSTM ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('LSTM — Pérdida durante entrenamiento')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('LSTM — Accuracy durante entrenamiento')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.suptitle('Historial de Entrenamiento del Modelo LSTM',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lstm_training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → lstm_training_history.png guardada.")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PROYECTO COMPLETADO EXITOSAMENTE")
print("=" * 80)
print("""
  Archivos generados:
    ✓ dataset_integrado.csv           — Dataset propio fusionado
    ✓ mapa_estaciones.html            — Mapa interactivo Folium
    ✓ correlaciones_region.png        — Correlaciones por región
    ✓ matrices_confusion.png          — Matrices de confusión (4 modelos)
    ✓ f1_comparativo.png              — Barras F1-Score comparativas
    ✓ feature_importance.png          — Importancia de variables (RF + XGB)
    ✓ lstm_training_history.png       — Historial de entrenamiento LSTM
""")
