import subprocess
import sys

PASOS = [
    ("1. Preparación de Datos (prepare.py)", [sys.executable, "src/prepare.py"]),
    ("2. Validación de Datos sin librerías externas (validate_data.py)", [sys.executable, "src/validate_data.py"]),
    ("3. Entrenamiento del Modelo con MLflow (train.py)", [sys.executable, "src/train.py"]),
    ("4. Pruebas Automatizadas del Modelo y Datos (pytest)", [sys.executable, "-m", "pytest", "tests/", "-v"]),
]

print("=" * 65)
print("EJECUTANDO PIPELINE AUTOMATIZADO DE MLOPS (UN SOLO COMANDO)")
print("=" * 65)

for nombre, comando in PASOS:
    print(f"\n>>> Etapa: {nombre}")
    res = subprocess.run(comando)
    if res.returncode != 0:
        print(f"[ERROR] Falló la etapa: {nombre}")
        sys.exit(res.returncode)

print("\n" + "=" * 65)
print("PIPELINE COMPLETADO Y VALIDADO CON ÉXITO")
print("=" * 65)
