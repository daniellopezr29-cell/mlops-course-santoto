# Pipeline MLOps Reproducible - Clasificación Iris (Semana 2)

**Especialización en Ciencia de Datos - Universidad Santo Tomás**
**Autor:** Daniel Ricardo López Rodríguez

## 1. Estructura del Proyecto
- `run_pipeline.py` / `Makefile` / `dvc.yaml`: Automatización del flujo completo con un solo comando.
- `src/prepare.py`: Preparación y división reproducible (80/20) del dataset Iris.
- `src/validate_data.py`: Validación nativa en Python (sin librerías externas) de tipos, ausencia de nulos y rangos lógicos.
- `data_contract.json`: Contrato de Datos oficial para inferencia (incluye ejemplo válido e inválido).
- `src/train.py`: Entrenamiento del modelo Random Forest (`n_estimators=50, max_depth=3`), registro en MLflow y serialización en `models/model.pkl`.
- `tests/test_model.py`: Pruebas unitarias automatizadas con `pytest` (`accuracy > 0.8`).

## 2. Instrucciones de Reproducción desde Cero
1. Clonar el repositorio:
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>

2. Crear entorno virtual e instalar dependencias:
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. Ejecutar el pipeline completo (preparación, validación, entrenamiento y testing) con un solo comando:
   python run_pipeline.py
   # Alternativamente: make all  o  dvc repro
