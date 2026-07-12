# B4_T2_XAI — Concesión de crédito sensible al coste + auditoría XAI

Práctica de máster (XAI). Se construye, **optimiza bajo coste** y **audita** un modelo
de concesión de crédito sobre el dataset *Give Me Some Credit* (`SeriousDlqin2yrs` = el
cliente entró en mora grave ≥90 días en 2 años).

Se entrega bajo **dos matrices de coste** distintas:

| Fichero | Coste FP | Coste FN | Interpretación |
|---|---|---|---|
| `cs_produccion1.csv` | 1 | 1 | los dos errores pesan igual |
| `cs_produccion2.csv` | 1 | 10 | fallar un moroso cuesta 10× denegar a un cliente bueno |

- **FP** = predecir "moroso" a un cliente bueno → le denegamos crédito sin motivo.
- **FN** = predecir "bueno" a un moroso → le concedemos crédito y no paga.

**Evaluación:** 50 % coste promedio en producción · 50 % calidad de la auditoría XAI.

---

## Cómo leer esta práctica (de principio a fin)

Léela en este orden; cada pieza depende de la anterior.

1. **`config.py` — el mapa.** Semilla global (`SEED=42`), rutas y las dos matrices de
   coste, todo en un sitio. Empieza aquí para entender dónde está cada cosa.

2. **`data_prep.py` — los datos.** Explica QUÉ limpiamos y POR QUÉ (renta ausente,
   centinelas 96/98 en los retrasos, outliers de ratios, edades imposibles). Clave:
   los parámetros de limpieza se aprenden **solo en train** y se aplican igual a val,
   test y producción → sin fuga de información y reproducible. Genera `data/processed/`.

3. **`baselines.py` — los modelos.** Cinco baselines (de interpretable a caja negra:
   logística, árbol, random forest, gradient boosting, LightGBM). No balanceamos las
   clases aquí a propósito: el desbalanceo se resuelve en el paso siguiente con el umbral.

4. **`cost_sensitive.py` — el corazón de la nota de "resultados".** Aquí está la idea
   importante: como el coste de los dos errores es distinto, **no** predecimos la clase
   más probable, sino que **movemos el umbral** de decisión para minimizar el coste
   esperado. Elegimos el umbral en validación, medimos el coste honesto en test y
   generamos los dos CSV. Incluye una tabla comparativa de coste por modelo.

5. **`shap_audit.py` — auditoría de Alonso (SHAP).** Explica la caja negra: qué variables
   pesan y en qué dirección (global) y por qué se decide cada caso concreto (local,
   waterfall). Es la parte de "por qué el modelo dice lo que dice".

6. **`surrogate_audit.py` (Raúl) y `counterfactual_audit.py` (Pietro)** — los otros dos
   flujos de auditoría (ver `REPARTO_TAREAS.md`).

7. **`notebook_entrega.ipynb` — el documento final.** Junta todo lo anterior con
   interpretación escrita después de cada resultado y una reflexión final. **Es el
   entregable.** Cada resultado numérico o gráfico va seguido de una celda markdown que
   explica qué significa.

> **Regla de oro del código:** todo está comentado en español explicando el *qué* y,
> sobre todo, el *por qué*. La intención es poder **entenderlo**, no solo ejecutarlo.

---

## Cómo ejecutarlo

Requisitos: `pandas numpy scikit-learn matplotlib lightgbm shap joblib`.

```bash
# 1) Deja los datos del profesor en data/raw/ (cs_construccion.csv, cs_produccion.csv, DataDictionary.csv)
python data_prep.py        # -> data/processed/
python baselines.py        # -> models/
python cost_sensitive.py   # -> cs_produccion1.csv, cs_produccion2.csv + tablas de coste
python shap_audit.py       # -> outputs/figures/ (SHAP)
```

Luego abrir `notebook_entrega.ipynb` y ejecutarlo de principio a fin (usa las funciones
de los módulos anteriores; no reentrena nada).

## Estructura del repositorio

```
config.py                 Configuración común (semilla, rutas, matrices de coste)
data_prep.py              Limpieza + splits train/val/test + producción
baselines.py              Entrenamiento y persistencia de los 5 baselines
cost_sensitive.py         Umbral óptimo por coste + tablas + CSV de producción
shap_audit.py             [Alonso] Auditoría SHAP global y local
surrogate_audit.py        [Raúl]  (pendiente) árbol subrogado + PDP/ICE + perm. importance
counterfactual_audit.py   [Pietro](pendiente) contrafactuals + explicación al cliente
notebook_entrega.ipynb    Entregable: consolida todo con interpretación
REPARTO_TAREAS.md         Reparto de tareas del grupo
data/  models/  outputs/  Datos, modelos y figuras generadas
```

**Grupo:** Alonso · Raúl · Pietro
