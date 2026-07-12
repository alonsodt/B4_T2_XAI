"""
baselines.py
============
Entrenamiento y persistencia de los modelos baseline (BASE COMÚN del grupo).

Modelos (de más interpretable a más "caja negra"):
  - logreg  : Regresión logística (dentro de un Pipeline con StandardScaler,
              porque la logística SÍ necesita escalado; los árboles no).
  - tree    : Árbol de decisión (interpretable, referencia sencilla).
  - rf      : Random Forest.
  - gb      : Gradient Boosting de sklearn.
  - lgbm    : LightGBM (suele ser el más fuerte en tabular).

POR QUÉ NO usamos class_weight/balanceo aquí:
  El desbalanceo (6,7 % de mora) lo resolvemos en la fase SENSIBLE AL COSTE
  (cost_sensitive.py) eligiendo el UMBRAL óptimo sobre la probabilidad. Eso es
  lo correcto teóricamente: entrenamos para estimar bien P(mora) y luego decidimos
  el corte según la matriz de coste. Así el MISMO modelo sirve para las dos
  matrices, solo cambia el umbral.

Todos los modelos llevan random_state fijo -> reproducibles.

Uso:
    python baselines.py            # entrena y guarda models/*.joblib
    from baselines import load_baseline
    modelo = load_baseline("lgbm")
"""

import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

import config
from data_prep import cargar_splits


def construir_modelos(seed=config.SEED) -> dict:
    """Devuelve el diccionario nombre -> estimador SIN entrenar.

    Los hiperparámetros son razonables y conservadores (no hacemos una búsqueda
    exhaustiva: el objetivo de la práctica es la AUDITORÍA XAI, no exprimir la
    última décima de AUC). max_depth limitado en árboles para evitar sobreajuste.
    """
    return {
        # Pipeline: escalado + logística. Guardar el Pipeline entero hace que el
        # escalado viaje CON el modelo (no hay que recordar escalar en predicción).
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=8, n_jobs=-1, random_state=seed
        ),
        "gb": GradientBoostingClassifier(random_state=seed),
        "lgbm": LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            random_state=seed, n_jobs=-1, verbose=-1,
        ),
    }


def entrenar_y_guardar(seed=config.SEED) -> dict:
    """Entrena todos los baselines, los guarda en models/ y devuelve un resumen.

    Reporta AUC en validación (métrica independiente del umbral, útil para ver
    qué modelo separa mejor las clases antes de meter el coste).
    """
    config.fijar_semillas(seed)
    config.crear_directorios()
    d = cargar_splits()
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]

    modelos = construir_modelos(seed)
    resumen = {}
    for nombre, modelo in modelos.items():
        t0 = time.time()
        modelo.fit(X_train, y_train)
        prob_val = modelo.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, prob_val)
        joblib.dump(modelo, config.MODELS_DIR / f"{nombre}.joblib")
        resumen[nombre] = {"auc_val": auc, "segundos": time.time() - t0}
        print(f"  {nombre:8s} AUC_val={auc:.4f}  ({resumen[nombre]['segundos']:.1f}s)  -> models/{nombre}.joblib")

    return resumen


def load_baseline(nombre: str):
    """Carga un modelo ya entrenado desde models/<nombre>.joblib.

    Es la función que usan el framework de coste y los tres flujos XAI para no
    reentrenar: todos comparten EXACTAMENTE el mismo objeto modelo.
    """
    ruta = config.MODELS_DIR / f"{nombre}.joblib"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero: python baselines.py"
        )
    return joblib.load(ruta)


def nombres_baselines() -> list:
    """Lista de nombres de baseline disponibles (orden estable)."""
    return list(construir_modelos().keys())


if __name__ == "__main__":
    print("Entrenando baselines (semilla fija = %d)..." % config.SEED)
    entrenar_y_guardar()
    print("\nHecho. Modelos en models/. Cárgalos con load_baseline('nombre').")
