"""
cost_sensitive.py
=================
Framework SENSIBLE AL COSTE (BASE COMÚN del grupo).

Idea central (el POR QUÉ de todo):
  El enunciado no evalúa por accuracy sino por COSTE PROMEDIO. Y el coste de los
  dos tipos de error es distinto:
     - Falso Positivo (FP): decimos "moroso" a un cliente bueno -> le denegamos
       crédito sin motivo. Coste c_fp.
     - Falso Negativo (FN): decimos "bueno" a un moroso -> le damos crédito y no
       paga. Coste c_fn.
  Con la matriz A (1,1) los dos errores pesan igual. Con la matriz B (1,10) fallar
  un moroso es 10 veces peor. Por tanto NO debemos predecir siempre la clase más
  probable: debemos mover el UMBRAL de decisión para minimizar el coste esperado.

Cómo elegimos el umbral:
  1. El modelo da una probabilidad P(mora) para cada cliente.
  2. Recorremos muchos umbrales candidatos sobre VALIDACIÓN y nos quedamos con el
     que minimiza el coste promedio en validación (datos que el modelo no vio).
  3. Ese umbral fijo se aplica luego a TEST (para estimar el coste honesto) y a
     PRODUCCIÓN (para generar las predicciones a entregar).

Umbral teórico de referencia: t* = c_fp / (c_fp + c_fn). Para A t*=0.5; para B
t*=1/11≈0.091 (bajamos el listón para "cazar" más morosos porque fallarlos es caro).
El umbral empírico que buscamos suele quedar cerca de este t* teórico.
"""

import numpy as np
import pandas as pd

import config
from data_prep import cargar_splits
from baselines import load_baseline, nombres_baselines


def coste_medio(y_true, y_pred, c_fp, c_fn) -> float:
    """Coste PROMEDIO por cliente de una predicción dada.

    Solo penalizan los errores: TP y TN cuestan 0. Dividimos por N para que sea
    'coste por cliente' (comparable entre datasets de distinto tamaño).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return (fp * c_fp + fn * c_fn) / len(y_true)


def optimizar_umbral(y_true, prob, c_fp, c_fn):
    """Busca sobre 'prob' el umbral que minimiza el coste promedio.

    Recorre como candidatos los propios valores de probabilidad observados
    (más un 0 y un 1.01 para cubrir "predecir todo 1" y "predecir todo 0").
    Devuelve (umbral_optimo, coste_en_este_conjunto).
    """
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)

    candidatos = np.unique(np.concatenate([[0.0], prob, [1.0 + 1e-9]]))
    mejor_umbral, mejor_coste = 0.5, np.inf
    for t in candidatos:
        y_pred = (prob >= t).astype(int)
        coste = coste_medio(y_true, y_pred, c_fp, c_fn)
        if coste < mejor_coste:
            mejor_coste, mejor_umbral = coste, t
    return float(mejor_umbral), float(mejor_coste)


def tabla_comparativa(matriz_nombre: str, c_fp: int, c_fn: int, splits=None) -> pd.DataFrame:
    """Compara TODOS los baselines para una matriz de coste concreta.

    Para cada modelo: ajusta el umbral en VAL y mide el coste en VAL y en TEST.
    El coste de TEST es el criterio honesto de selección (val se usó para elegir
    el umbral, así que no sería justo seleccionar por val).
    """
    if splits is None:
        splits = cargar_splits()
    y_val, y_test = splits["y_val"], splits["y_test"]

    filas = []
    for nombre in nombres_baselines():
        modelo = load_baseline(nombre)
        prob_val = modelo.predict_proba(splits["X_val"])[:, 1]
        prob_test = modelo.predict_proba(splits["X_test"])[:, 1]

        umbral, coste_val = optimizar_umbral(y_val, prob_val, c_fp, c_fn)
        coste_test = coste_medio(y_test, (prob_test >= umbral).astype(int), c_fp, c_fn)

        filas.append({
            "modelo": nombre,
            "umbral_opt": round(umbral, 4),
            "coste_val": round(coste_val, 5),
            "coste_test": round(coste_test, 5),
        })

    tabla = pd.DataFrame(filas).sort_values("coste_test").reset_index(drop=True)
    tabla.insert(0, "matriz", matriz_nombre)
    return tabla


def coste_baseline_trivial(y_true, c_fp, c_fn):
    """Coste de las dos estrategias tontas, como referencia.

    - 'todo 0' (conceder a todos): solo hay FN -> coste = %mora * c_fn.
    - 'todo 1' (denegar a todos): solo hay FP -> coste = %buenos * c_fp.
    El modelo debe batir claramente a la mejor de estas dos.
    """
    y_true = np.asarray(y_true)
    coste_todo0 = coste_medio(y_true, np.zeros_like(y_true), c_fp, c_fn)
    coste_todo1 = coste_medio(y_true, np.ones_like(y_true), c_fp, c_fn)
    return {"conceder_a_todos": coste_todo0, "denegar_a_todos": coste_todo1}


def generar_produccion(modelo_nombre: str, umbral: float, salida) -> pd.DataFrame:
    """Predice sobre producción con un modelo+umbral y guarda el CSV a entregar.

    El CSV tiene una única columna 'SeriousDlqin2yrs' con la predicción 0/1,
    una fila por cliente de producción y en el MISMO orden que cs_produccion.csv.
    """
    splits = cargar_splits()
    modelo = load_baseline(modelo_nombre)
    prob_prod = modelo.predict_proba(splits["X_prod"])[:, 1]
    pred = (prob_prod >= umbral).astype(int)

    df_out = pd.DataFrame({config.TARGET: pred})
    df_out.to_csv(salida, index=False)
    return df_out


def ejecutar_todo():
    """Pipeline completo de coste: tablas de ambas matrices + genera los 2 CSV.

    Devuelve (tablas, seleccion) para poder mostrarlo en el notebook.
    """
    splits = cargar_splits()
    tablas = {}
    seleccion = {}

    for matriz_nombre, cfg in config.MATRICES_COSTE.items():
        c_fp, c_fn = cfg["c_fp"], cfg["c_fn"]
        tabla = tabla_comparativa(matriz_nombre, c_fp, c_fn, splits)
        tablas[matriz_nombre] = tabla

        # El mejor modelo es el de menor coste en TEST (primera fila tras ordenar).
        mejor = tabla.iloc[0]
        seleccion[matriz_nombre] = {
            "modelo": mejor["modelo"],
            "umbral": float(mejor["umbral_opt"]),
            "coste_test": float(mejor["coste_test"]),
            "trivial": coste_baseline_trivial(splits["y_test"], c_fp, c_fn),
        }

        # Generamos el CSV de producción con ese mejor modelo y su umbral.
        generar_produccion(mejor["modelo"], mejor["umbral_opt"], cfg["salida"])

    return tablas, seleccion


if __name__ == "__main__":
    config.fijar_semillas()
    tablas, seleccion = ejecutar_todo()
    for matriz_nombre, tabla in tablas.items():
        cfg = config.MATRICES_COSTE[matriz_nombre]
        print(f"\n=== Matriz {matriz_nombre}  (FP={cfg['c_fp']}, FN={cfg['c_fn']}) ===")
        print(tabla.to_string(index=False))
        sel = seleccion[matriz_nombre]
        print(f"  -> MEJOR: {sel['modelo']} (umbral={sel['umbral']:.4f}, coste_test={sel['coste_test']:.5f})")
        print(f"     referencia trivial: {sel['trivial']}")
        print(f"     CSV escrito en: {cfg['salida'].name}")
