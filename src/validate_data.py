import csv
import json
import math
import os
import sys

# Rangos lógicos biológicos del dataset Iris
RANGOS_CSV = {
    "sepal length (cm)": (4.0, 8.0),
    "sepal width (cm)": (2.0, 4.5),
    "petal length (cm)": (1.0, 7.0),
    "petal width (cm)": (0.1, 2.5),
}

RANGOS_JSON = {
    "sepal_length": (4.0, 8.0),
    "sepal_width": (2.0, 4.5),
    "petal_length": (1.0, 7.0),
    "petal_width": (0.1, 2.5),
}

ESPECIES_VALIDAS = {0, 1, 2}

def es_nulo(valor) -> bool:
    if valor is None:
        return True
    if isinstance(valor, str) and valor.strip().lower() in ("", "null", "none", "nan"):
        return True
    if isinstance(valor, float) and (math.isnan(valor) or math.isinf(valor)):
        return True
    return False

def es_numero_valido(valor) -> bool:
    if isinstance(valor, bool):
        return False
    return isinstance(valor, (int, float))

def validar_payload_json(payload: dict) -> tuple:
    errores = []
    for campo, (min_v, max_v) in RANGOS_JSON.items():
        if campo not in payload:
            errores.append(f"Campo faltante: '{campo}'.")
            continue
        val = payload[campo]
        if es_nulo(val):
            errores.append(f"Valor nulo detectado en '{campo}'.")
            continue
        if not es_numero_valido(val):
            errores.append(f"Tipo de dato inválido en '{campo}': se recibió {type(val).__name__}.")
            continue
        if not (min_v <= float(val) <= max_v):
            errores.append(f"Fuera de rango lógico en '{campo}': {val} no está en [{min_v}, {max_v}].")
    return len(errores) == 0, errores

def validar_archivo_csv(ruta: str) -> tuple:
    errores = []
    if not os.path.exists(ruta):
        return False, [f"No existe el archivo: {ruta}"]

    with open(ruta, mode="r", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila_idx, fila in enumerate(lector, start=2):
            for col, (min_v, max_v) in RANGOS_CSV.items():
                raw = fila.get(col)
                if es_nulo(raw):
                    errores.append(f"Fila {fila_idx}: Valor nulo en '{col}'.")
                    continue
                try:
                    num = float(raw)
                except (ValueError, TypeError):
                    errores.append(f"Fila {fila_idx}: Tipo inválido en '{col}' ({raw}).")
                    continue
                if not (min_v <= num <= max_v):
                    errores.append(f"Fila {fila_idx}: '{col}'={num} fuera de [{min_v}, {max_v}].")

            sp = fila.get("species")
            if es_nulo(sp):
                errores.append(f"Fila {fila_idx}: Valor nulo en 'species'.")
            else:
                try:
                    sp_num = float(sp)
                    if not sp_num.is_integer() or int(sp_num) not in ESPECIES_VALIDAS:
                        errores.append(f"Fila {fila_idx}: 'species'={sp} fuera de {ESPECIES_VALIDAS}.")
                except (ValueError, TypeError):
                    errores.append(f"Fila {fila_idx}: Tipo inválido en 'species' ({sp}).")
    return len(errores) == 0, errores

if __name__ == "__main__":
    # 1. Validar CSVs de entrenamiento y prueba
    for ruta_csv in ["data/processed/train.csv", "data/processed/test.csv"]:
        ok, errs = validar_archivo_csv(ruta_csv)
        if not ok:
            print(f"[ERROR] Falló validación en {ruta_csv}: {errs[:5]}")
            sys.exit(1)
        print(f"[OK] {ruta_csv} validado (tipos correctos, sin nulos y en rangos lógicos).")

    # 2. Verificar el Contrato de Datos JSON
    with open("data_contract.json", "r", encoding="utf-8") as f:
        contrato = json.load(f)

    ej_valido = contrato["examples"]["valid_input"]["payload"]
    ej_invalido = contrato["examples"]["invalid_input"]["payload"]

    ok_v, errs_v = validar_payload_json(ej_valido)
    assert ok_v, f"Falló ejemplo válido: {errs_v}"
    print("[OK] Contrato JSON: Ejemplo de entrada válida aprobado.")

    ok_inv, errs_inv = validar_payload_json(ej_invalido)
    assert not ok_inv, "El ejemplo inválido no fue detectado."
    print(f"[OK] Contrato JSON: Entrada inválida rechazada correctamente ({len(errs_inv)} errores detectados):")
    for e in errs_inv:
        print(f"     - {e}")
