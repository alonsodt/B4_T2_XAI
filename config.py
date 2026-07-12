"""
config.py
=========
Configuración central y compartida por TODO el proyecto (base común del grupo).

POR QUÉ este fichero:
- Fijar UNA sola semilla global y unas rutas únicas evita que cada persona del
  grupo (Alonso, Raúl, Pietro) use valores distintos. Así los tres cargamos
  EXACTAMENTE los mismos splits y modelos y los resultados son reproducibles.
- Si mañana cambiamos una ruta o la semilla, se cambia aquí y punto.
"""

from pathlib import Path
import os
import random
import numpy as np

# --- Semilla global -----------------------------------------------------------
# La usamos en el split, en cada modelo y en cualquier muestreo. Es la clave de
# la reproducibilidad: mismo SEED -> mismos resultados en cualquier máquina.
SEED = 42


def fijar_semillas(seed: int = SEED) -> None:
    """Fija la semilla en las tres fuentes de aleatoriedad que usamos.

    POR QUÉ: numpy (sklearn/shap por debajo), el módulo random de Python y la
    variable de entorno PYTHONHASHSEED. Llamar a esto al empezar cualquier
    script/notebook garantiza que dos ejecuciones den el mismo número.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# --- Rutas del proyecto -------------------------------------------------------
# Todo relativo a la raíz del repo para que funcione igual en Windows/Linux/Colab.
ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"          # figuras y tablas generadas
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Ficheros de entrada (los deja el profesor / el grupo en data/raw/)
RAW_CONSTRUCCION = DATA_RAW / "cs_construccion.csv"   # tiene target -> para entrenar/validar/test
RAW_PRODUCCION = DATA_RAW / "cs_produccion.csv"       # target vacío  -> lo que hay que predecir
RAW_DICCIONARIO = DATA_RAW / "DataDictionary.csv"

# Ficheros procesados (los genera data_prep.py; los cargan los tres flujos XAI)
TRAIN_CSV = DATA_PROCESSED / "train.csv"
VAL_CSV = DATA_PROCESSED / "val.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"
PRODUCCION_CSV = DATA_PROCESSED / "produccion.csv"        # producción ya limpia (sin target)
PREPROCESS_PARAMS = DATA_PROCESSED / "preprocess_params.json"

# Entregables de producción (predicciones que evalúa el profesor por coste)
CS_PRODUCCION_1 = ROOT / "cs_produccion1.csv"   # matriz de coste A: FP=1, FN=1
CS_PRODUCCION_2 = ROOT / "cs_produccion2.csv"   # matriz de coste B: FP=1, FN=10

# --- Nombres de columnas ------------------------------------------------------
TARGET = "SeriousDlqin2yrs"   # 1 = el cliente entró en mora grave (>=90 días) en 2 años

# Las dos matrices de coste que pide el enunciado.
# clave: (coste_falso_positivo, coste_falso_negativo)
#   FP = predecimos mora (1) pero el cliente era bueno (0)  -> denegamos crédito a un buen cliente
#   FN = predecimos bueno (0) pero el cliente entra en mora (1) -> concedemos crédito a un moroso
MATRICES_COSTE = {
    "produccion1": {"c_fp": 1, "c_fn": 1, "salida": CS_PRODUCCION_1},
    "produccion2": {"c_fp": 1, "c_fn": 10, "salida": CS_PRODUCCION_2},
}


def crear_directorios() -> None:
    """Crea las carpetas de salida si no existen (idempotente)."""
    for d in (DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
