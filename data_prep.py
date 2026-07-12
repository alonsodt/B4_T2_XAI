"""
data_prep.py
============
Carga, limpieza y particionado de los datos (BASE COMÚN del grupo).

Objetivo: que Alonso, Raúl y Pietro carguen EXACTAMENTE los mismos datos.
Para ello:
  1. Limpiamos con una lógica única y documentada.
  2. Los parámetros de limpieza (medianas, topes de recorte) se APRENDEN solo
     en train y se guardan en un JSON. Val, test y producción se transforman
     con esos MISMOS parámetros -> no hay fuga de información (data leakage) y
     la transformación es idéntica y reproducible en cualquier máquina.
  3. Partimos construcción en train/val/test de forma ESTRATIFICADA (mantiene
     el 6,7 % de morosos en los tres) y fijamos la semilla.

Uso:
    python data_prep.py          # genera data/processed/*.csv y preprocess_params.json

Decisiones de limpieza (el QUÉ y el POR QUÉ):
  - age: hay algún 0 imposible. Recortamos a [18, 100] (un menor no pide crédito;
    >100 es error). Casi no afecta (1 caso) pero deja la variable sana.
  - MonthlyIncome: ~20 % de nulos. NO borramos filas porque en producción hay que
    predecir TODAS. Imputamos la mediana y añadimos un flag `MonthlyIncome_missing`
    porque el hecho de que falte la renta es informativo (correlaciona con la mora).
  - NumberOfDependents: ~2,6 % de nulos -> imputamos la mediana (que es 0).
  - Columnas de retrasos (30-59, 60-89, 90 días): valores 96 y 98 son códigos
    centinela de "no disponible", no un número real de retrasos. Los tratamos como
    nulos y los imputamos con la mediana de la columna.
  - RevolvingUtilization, DebtRatio, MonthlyIncome: tienen outliers enormes
    (ratios de 20.000, rentas de 3 M). Recortamos (winsorizamos) por arriba al
    percentil 99,5 de train. Esto estabiliza sobre todo a la regresión logística.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

# Columnas cuyos valores 96/98 son centinelas de "dato no disponible".
COLS_RETRASOS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
# Columnas continuas muy sesgadas que winsorizamos por la cola alta.
COLS_WINSOR = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]
SENTINELA_RETRASO = 90        # cualquier valor >= 90 (96, 98) se considera centinela
PCT_WINSOR = 0.995            # percentil de recorte superior


def ajustar_parametros(df_train: pd.DataFrame) -> dict:
    """Aprende SOLO en train los parámetros de limpieza y los devuelve como dict.

    POR QUÉ solo en train: si calculásemos las medianas/topes usando val o test
    estaríamos filtrando información del futuro al modelo (data leakage) y el
    coste estimado sería optimista y no reproducible.
    """
    params = {"medianas": {}, "topes_winsor": {}}

    # Trabajamos sobre una copia ya "sentinela->NaN" para que las medianas no se
    # contaminen con los 96/98.
    tr = df_train.copy()
    for c in COLS_RETRASOS:
        tr.loc[tr[c] >= SENTINELA_RETRASO, c] = np.nan

    # age fuera de rango tampoco debe influir en su mediana
    tr.loc[(tr["age"] < 18) | (tr["age"] > 100), "age"] = np.nan

    # Medianas para imputar (todas las columnas de features)
    features = [c for c in df_train.columns if c != config.TARGET]
    for c in features:
        params["medianas"][c] = float(tr[c].median())

    # Topes de winsorizado (percentil 99,5 de train, ya sin outliers de renta absurdos)
    for c in COLS_WINSOR:
        params["topes_winsor"][c] = float(tr[c].quantile(PCT_WINSOR))

    return params


def aplicar_limpieza(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Aplica la MISMA limpieza a cualquier split usando los parámetros de train.

    Devuelve un DataFrame con las features limpias (+ flag de renta) y, si existe,
    la columna target intacta.
    """
    out = df.copy()

    # 1) Flag de renta ausente ANTES de imputar (si no, se pierde la información).
    out["MonthlyIncome_missing"] = out["MonthlyIncome"].isna().astype(int)

    # 2) age a rango plausible -> fuera de [18,100] se trata como nulo.
    out.loc[(out["age"] < 18) | (out["age"] > 100), "age"] = np.nan

    # 3) Centinelas 96/98 en columnas de retrasos -> nulo.
    for c in COLS_RETRASOS:
        out.loc[out[c] >= SENTINELA_RETRASO, c] = np.nan

    # 4) Winsorizado superior de continuas sesgadas (recorte, no borrado).
    for c, tope in params["topes_winsor"].items():
        out[c] = out[c].clip(upper=tope)

    # 5) Imputación de nulos con la mediana de train (columnas de features).
    for c, med in params["medianas"].items():
        if c in out.columns:
            out[c] = out[c].fillna(med)

    return out


def cargar_y_particionar(test_size=0.2, val_size=0.2, seed=config.SEED):
    """Carga construcción, hace el split estratificado y limpia cada parte.

    Split: primero separamos test (20 %); del resto separamos val (20 % del resto).
    Resultado aproximado: 64 % train / 16 % val / 20 % test. Estratificado por
    target para conservar el ~6,7 % de morosos en los tres.
    """
    config.fijar_semillas(seed)

    df = pd.read_csv(config.RAW_CONSTRUCCION)

    # Separamos features/target manteniendo el DataFrame junto para el split.
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[config.TARGET]
    )
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_size, random_state=seed,
        stratify=df_trainval[config.TARGET],
    )

    # Aprendemos parámetros SOLO en train y limpiamos las tres particiones igual.
    params = ajustar_parametros(df_train)
    df_train = aplicar_limpieza(df_train, params)
    df_val = aplicar_limpieza(df_val, params)
    df_test = aplicar_limpieza(df_test, params)

    return df_train, df_val, df_test, params


def preparar_produccion(params: dict) -> pd.DataFrame:
    """Limpia el fichero de producción con los MISMOS parámetros de train.

    Producción tiene el target vacío; devolvemos solo las features (mismo orden
    de filas que el CSV original, porque las predicciones deben ir alineadas).
    """
    prod = pd.read_csv(config.RAW_PRODUCCION)
    prod = prod.drop(columns=[config.TARGET])          # target vacío, lo quitamos
    prod_limpio = aplicar_limpieza(prod, params)
    return prod_limpio


def main():
    config.crear_directorios()
    df_train, df_val, df_test, params = cargar_y_particionar()

    # Guardamos los parámetros de limpieza (auditable y reutilizable).
    with open(config.PREPROCESS_PARAMS, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    # Guardamos las particiones ya limpias.
    df_train.to_csv(config.TRAIN_CSV, index=False)
    df_val.to_csv(config.VAL_CSV, index=False)
    df_test.to_csv(config.TEST_CSV, index=False)

    # Producción limpia (para que los tres flujos predigan sobre lo mismo).
    prod_limpio = preparar_produccion(params)
    prod_limpio.to_csv(config.PRODUCCION_CSV, index=False)

    # Resumen por pantalla (control de calidad rápido).
    def resumen(nombre, d):
        tasa = 100 * d[config.TARGET].mean() if config.TARGET in d else float("nan")
        print(f"  {nombre:12s} filas={len(d):6d}  % mora={tasa:5.2f}  nulos={int(d.isna().sum().sum())}")

    print("Datos procesados y guardados en data/processed/:")
    resumen("train", df_train)
    resumen("val", df_val)
    resumen("test", df_test)
    print(f"  {'produccion':12s} filas={len(prod_limpio):6d}  (sin target)      nulos={int(prod_limpio.isna().sum().sum())}")
    print(f"\nColumnas finales ({len(prod_limpio.columns)}): {list(prod_limpio.columns)}")


# ---- API de carga para el resto del grupo -----------------------------------
def cargar_splits():
    """Carga train/val/test ya procesados como (X, y) cada uno.

    Es la función que usan baselines.py y los tres flujos XAI para no repetir
    la lógica de carga. Devuelve un dict con X_train,y_train,... y X_prod.
    """
    train = pd.read_csv(config.TRAIN_CSV)
    val = pd.read_csv(config.VAL_CSV)
    test = pd.read_csv(config.TEST_CSV)
    prod = pd.read_csv(config.PRODUCCION_CSV)

    def xy(d):
        return d.drop(columns=[config.TARGET]), d[config.TARGET]

    X_train, y_train = xy(train)
    X_val, y_val = xy(val)
    X_test, y_test = xy(test)
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "X_prod": prod,
    }


if __name__ == "__main__":
    main()
