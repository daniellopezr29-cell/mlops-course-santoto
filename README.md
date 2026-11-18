# Pipeline MLOps Reproducible - Clasificación Iris (Semanas 1 y 2)

**Especialización en Ciencia de Datos - Universidad Santo Tomás (Seminario de Grado II)**  
**Autor:** Daniel Ricardo López Rodríguez  

Este repositorio implementa un flujo de **MLOps** reproducible para el entrenamiento, validación de datos, control de calidad y registro de un clasificador `RandomForestClassifier` sobre el dataset Iris, integrando **MLflow**, **DVC**, validación nativa de Contratos de Datos y pruebas automatizadas con **pytest**.

---

## 1. Arquitectura y Archivos del Proyecto

```text
mlops-course-santoto/
├── run_pipeline.py             # Orquestador en un solo comando (python run_pipeline.py)
├── Makefile                    # Automatización por consola (make all)
├── dvc.yaml                    # Pipeline declarativo con DVC (prepare -> validate -> train -> test)
├── data_contract.json          # Contrato de Datos oficial en JSON (ejemplo válido e inválido)
├── requirements.txt            # Dependencias completas del stack MLOps
├── src/
│   ├── prepare.py              # Ingesta de Iris y división reproducible 80/20 (random_state=42)
│   ├── validate_data.py        # Validador en Python puro (sin librerías externas)
│   └── train.py                # Entrenamiento (n_estimators=50, max_depth=3), MLflow y models/model.pkl
├── tests/
│   └── test_model.py           # 5 pruebas automatizadas con pytest (accuracy > 0.80)
├── data/processed/             # Datasets generados (train.csv y test.csv)
└── models/                     # Modelo serializado (model.pkl)
