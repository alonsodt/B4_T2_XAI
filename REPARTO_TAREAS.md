# Reparto de tareas — Práctica B4-T2 XAI

**Grupo:** Alonso, Raúl, Pietro
**Entregable final:** un único `notebook_entrega.ipynb` + `cs_produccion1.csv` + `cs_produccion2.csv`
**Evaluación:** 50 % coste promedio en producción · 50 % calidad de la auditoría XAI

---

## Base común (la deja hecha Alonso — ya está)

Infraestructura que los tres cargamos igual para no pisarnos ni desincronizar datos.
Cada módulo se ejecuta como script (`python x.py`) y también expone funciones para el notebook.

| Fichero | Objetivo | Entradas | Salidas |
|---|---|---|---|
| `config.py` | Semilla (`SEED=42`), rutas y matrices de coste en un solo sitio. | — | Constantes + `fijar_semillas()`, `crear_directorios()` |
| `data_prep.py` | Limpieza + split estratificado train/val/test. Parámetros aprendidos SOLO en train (sin fuga). Misma transformación a producción. | `data/raw/cs_construccion.csv`, `cs_produccion.csv` | `data/processed/{train,val,test,produccion}.csv`, `preprocess_params.json`; función `cargar_splits()` |
| `baselines.py` | Entrena 5 baselines (logreg, tree, rf, gb, lgbm) con semilla fija. | splits procesados | `models/*.joblib`; función `load_baseline(nombre)` |
| `cost_sensitive.py` | Elige umbral óptimo por matriz de coste sobre validación, mide coste en test, tabla comparativa y genera los 2 CSV de producción. | modelos + splits | `cs_produccion1.csv`, `cs_produccion2.csv`; funciones `optimizar_umbral()`, `coste_medio()`, `tabla_comparativa()` |

**Cómo arrancar (los tres, una vez):**
```bash
python data_prep.py      # genera data/processed/
python baselines.py      # genera models/
python cost_sensitive.py # genera los 2 CSV y las tablas de coste
```

**Reproducibilidad:** todo pasa por `config.fijar_semillas(42)`; los splits y modelos guardados
son idénticos byte a byte para los tres. Nadie reentrena: se usa `load_baseline()`.

---

## Auditoría XAI (el otro 50 %) — 3 flujos INDEPENDIENTES

Cada flujo es su propio módulo, carga los mismos splits/modelos y puede desarrollarse
por separado. Al final se pegan en `notebook_entrega.ipynb`.

### Alonso — SHAP  ·  `shap_audit.py`  ·  **HECHO**
- **Objetivo:** explicar la caja negra (LightGBM) a nivel global y local con SHAP.
- **Entradas:** `load_baseline("lgbm")`, `X_test`.
- **Salidas:** `outputs/figures/shap_beeswarm.png`, `shap_bar.png`, `shap_dependence_*.png`,
  `shap_waterfall_{alto_riesgo,bajo_riesgo,frontera}.png`.
- **Contenido:** beeswarm + barras (importancia y signo global), dependence de las 2
  variables top, y waterfalls de 3 clientes (alto riesgo, bajo riesgo, frontera).
- **Esfuerzo:** ~medio (hecho).

### Raúl — Modelo subrogado + otras técnicas  ·  `surrogate_audit.py`  ·  **PENDIENTE**
- **Objetivo:** aproximar la caja negra con un árbol subrogado (caja blanca) y extraer
  reglas legibles; medir la FIDELIDAD del subrogado (¿cuánto coincide con la caja negra?).
  Añadir permutation importance y PDP/ICE como técnicas complementarias.
- **Entradas:** `load_baseline("lgbm")`, `X_train`/`X_test` de `cargar_splits()`.
- **Salidas sugeridas:** `outputs/figures/surrogate_tree.png`, `outputs/reglas_surrogate.txt`,
  `outputs/figures/permutation_importance.png`, `outputs/figures/pdp_*.png`; métrica de fidelidad
  (accuracy del árbol subrogado prediciendo la salida de la caja negra).
- **Cómo empezar:** entrenar `DecisionTreeClassifier(max_depth=3-4)` sobre `X` con etiqueta =
  `lgbm.predict(X)` (¡la predicción del modelo, no el target real!); exportar reglas con
  `sklearn.tree.export_text`. Permutation importance con `sklearn.inspection.permutation_importance`.
  PDP/ICE con `sklearn.inspection.PartialDependenceDisplay`.
- **Esfuerzo:** ~medio-alto.

### Pietro — Contrafactuals  ·  `counterfactual_audit.py`  ·  **PENDIENTE**
- **Objetivo:** para clientes reales de clase 0 y de clase 1, encontrar el cambio MÍNIMO
  en sus variables que voltearía la decisión ("si tuvieras 2 retrasos menos, te aprobamos").
  Redactar la explicación que se le daría a un cliente al que se le deniega el crédito.
- **Entradas:** `load_baseline("lgbm")`, umbral de `cost_sensitive` (p. ej. el de la matriz B),
  `X_test`, `y_test`.
- **Salidas sugeridas:** tabla de contrafactuals (variable cambiada, valor original → nuevo),
  `outputs/contrafactuals.csv`, y un párrafo de explicación al cliente.
- **Cómo empezar:** librería `dice-ml` (DiCE) sobre el modelo; o una búsqueda propia sencilla
  (perturbar variables accionables como utilización o nº de retrasos hasta cruzar el umbral).
  Ojo: usar solo variables ACCIONABLES (la edad no se puede cambiar).
- **Esfuerzo:** ~medio-alto.

---

## Consolidación final (entre los tres)
1. Cada uno deja su módulo funcionando y sus figuras en `outputs/`.
2. Se montan las secciones 5–7 del `notebook_entrega.ipynb` (una por flujo) importando
   cada módulo y comentando resultados.
3. Reflexión final conjunta: coste obtenido vs. lo que dicen las explicaciones (¿el modelo
   usa variables razonables?, ¿qué le diríamos a un cliente denegado?).
4. Revisar que el notebook corre de principio a fin sin errores y con la semilla fija.
