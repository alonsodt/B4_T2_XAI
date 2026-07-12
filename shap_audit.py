"""
shap_audit.py  ---  Flujo XAI de ALONSO: análisis SHAP
======================================================
Audita el modelo "caja negra" (LightGBM) con SHAP, en dos niveles:

  GLOBAL  -> ¿qué variables mueven el modelo en su conjunto y en qué dirección?
             - beeswarm (summary): importancia + signo de cada variable.
             - barras: importancia media (|SHAP|).
             - dependence: cómo cambia el efecto de UNA variable según su valor.

  LOCAL   -> ¿por qué el modelo decidió ESTO para ESTE cliente concreto?
             - waterfall: descompone una predicción individual en la suma de las
               contribuciones de cada variable partiendo del valor base.

Fundamento: SHAP reparte la predicción entre las variables usando los valores de
Shapley (teoría de juegos). Para modelos de árboles usamos TreeExplainer, que es
exacto y rápido. El valor SHAP está en escala de log-odds (margen) del modelo:
positivo = empuja hacia "mora (1)"; negativo = empuja hacia "buen cliente (0)".

Uso:
    python shap_audit.py          # calcula SHAP y guarda figuras en outputs/figures/
"""

import matplotlib
import matplotlib.pyplot as plt
# NOTA: no forzamos el backend al importar el módulo. Si se importa desde el
# notebook queremos que las figuras se muestren inline; solo cuando se ejecuta
# como script (abajo, en __main__) fijamos el backend "Agg" para poder GUARDAR
# figuras sin necesidad de una ventana gráfica.
import numpy as np
import shap

import config
from data_prep import cargar_splits
from baselines import load_baseline

# Modelo a auditar: el LightGBM (fuerte y, además, el mejor en la matriz B).
MODELO_AUDITAR = "lgbm"
# Nº de clientes de test que muestreamos para las figuras globales (legibilidad).
N_MUESTRA_GLOBAL = 2000


def calcular_shap(modelo_nombre=MODELO_AUDITAR, n_muestra=N_MUESTRA_GLOBAL, seed=config.SEED):
    """Calcula los valores SHAP sobre una muestra de test.

    Devuelve (explanation, X_muestra, modelo). 'explanation' es el objeto de SHAP
    con .values (contribuciones), .base_values (valor base) y .data (los X).
    """
    config.fijar_semillas(seed)
    splits = cargar_splits()
    X_test = splits["X_test"]

    # Muestreamos para que el beeswarm sea legible y el cálculo, ligero.
    X_muestra = X_test.sample(n=min(n_muestra, len(X_test)), random_state=seed)

    modelo = load_baseline(modelo_nombre)
    explainer = shap.TreeExplainer(modelo)
    explanation = explainer(X_muestra)

    # Para LightGBM binario, algunas versiones devuelven una dimensión extra por
    # clase. Nos quedamos con la contribución hacia la clase 1 (mora).
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]

    return explanation, X_muestra, modelo


def figuras_globales(explanation, X_muestra):
    """Genera y guarda las figuras SHAP globales."""
    config.crear_directorios()

    # 1) Beeswarm: cada punto es un cliente; color = valor de la variable.
    #    Muestra a la vez la importancia (orden) y el signo del efecto.
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "shap_beeswarm.png", dpi=130, bbox_inches="tight")
    plt.close()

    # 2) Barras: importancia media absoluta (ranking limpio de variables).
    plt.figure()
    shap.plots.bar(explanation, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "shap_bar.png", dpi=130, bbox_inches="tight")
    plt.close()

    # 3) Dependence de las 2 variables más importantes: efecto vs valor.
    orden = np.argsort(-np.abs(explanation.values).mean(axis=0))
    top_feats = [X_muestra.columns[i] for i in orden[:2]]
    for feat in top_feats:
        plt.figure()
        shap.plots.scatter(explanation[:, feat], show=False)
        plt.tight_layout()
        nombre = feat.replace("/", "_")[:40]
        plt.savefig(config.FIGURES_DIR / f"shap_dependence_{nombre}.png", dpi=130, bbox_inches="tight")
        plt.close()

    return top_feats


def seleccionar_casos_locales(modelo, seed=config.SEED):
    """Elige clientes concretos e interesantes para el análisis local.

    Buscamos 3 perfiles didácticos en test:
      - alto riesgo: el cliente con mayor P(mora) (el modelo lo deniega con fuerza).
      - bajo riesgo: el cliente con menor P(mora) (concesión clara).
      - frontera:    P(mora) cercana a 0.5 (caso dudoso).
    Devuelve un dict nombre -> índice posicional dentro de X_test.
    """
    splits = cargar_splits()
    X_test = splits["X_test"]
    prob = modelo.predict_proba(X_test)[:, 1]
    return {
        "alto_riesgo": int(np.argmax(prob)),
        "bajo_riesgo": int(np.argmin(prob)),
        "frontera": int(np.argmin(np.abs(prob - 0.5))),
    }, X_test, prob


def figuras_locales(modelo_nombre=MODELO_AUDITAR, seed=config.SEED):
    """Genera un waterfall por cada caso seleccionado y lo guarda."""
    config.crear_directorios()
    modelo = load_baseline(modelo_nombre)
    casos, X_test, prob = seleccionar_casos_locales(modelo, seed)

    explainer = shap.TreeExplainer(modelo)
    for etiqueta, idx in casos.items():
        fila = X_test.iloc[[idx]]
        exp = explainer(fila)
        if exp.values.ndim == 3:
            exp = exp[:, :, 1]

        plt.figure()
        shap.plots.waterfall(exp[0], max_display=12, show=False)
        plt.title(f"{etiqueta}  |  P(mora)={prob[idx]:.3f}")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / f"shap_waterfall_{etiqueta}.png", dpi=130, bbox_inches="tight")
        plt.close()

    return {k: {"idx": v, "prob": float(prob[v])} for k, v in casos.items()}


if __name__ == "__main__":
    matplotlib.use("Agg")  # sin ventana: guardamos las figuras a disco
    print(f"Calculando SHAP sobre el modelo '{MODELO_AUDITAR}'...")
    explanation, X_muestra, modelo = calcular_shap()
    top = figuras_globales(explanation, X_muestra)
    print(f"  Figuras globales guardadas. Top-2 variables: {top}")
    casos = figuras_locales()
    print("  Figuras locales (waterfall) guardadas para casos:")
    for k, v in casos.items():
        print(f"    {k:12s} idx_test={v['idx']:6d}  P(mora)={v['prob']:.3f}")
    print(f"\nTodo en {config.FIGURES_DIR}")
