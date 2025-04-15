
import pandas as pd

# === File di origine ===
data_dir = "NHANES/1988-2018/"
output_path = "lifemina_dataset.csv"


# === Specifica delle colonne per ciascun biomarcatore: (colonna, file sorgente) ===
column_choices = {
    "albumin": [("LBXSAL", "response_clean.csv")],
    "creatinine": [("LBXSCR", "response_clean.csv")],
    "glucose": [("LBXSGL", "response_clean.csv")],
    "log_crp": [("LBXHSCRP", "response_clean.csv")],
    "lymphocyte_percent": [("LBXLYPCT", "response_clean.csv")],
    "mean_cell_volume": [("LBXMCVSI", "response_clean.csv")],
    "red_cell_distribution_width": [("LBXRDW", "response_clean.csv")],
    "alkaline_phosphatase": [("LBXSAPSI", "response_clean.csv")],
    "white_blood_cell_count": [("LBXWBCSI", "response_clean.csv")],
    "age": [("RIDAGEYR", "demographics_clean.csv")]
}

# === Caricamento dei file necessari ===
files_needed = {}
for fields in column_choices.values():
    for var, file in fields:
        files_needed.setdefault(file, set()).add(var)
files_loaded = {fname: pd.read_csv(data_dir + fname, usecols=["SEQN"] + list(vars), low_memory=False)
                for fname, vars in files_needed.items()}

# === Merge progressivo su SEQN ===
from functools import reduce
df_merged = reduce(lambda left, right: pd.merge(left, right, on="SEQN", how="inner"), files_loaded.values())

# === Selezione del primo valore disponibile per ogni biomarcatore ===
def select_first_available(row, options):
    for col, _ in options:
        if col in row and pd.notna(row[col]):
            return row[col]
    return None

# Costruzione dataset finale
result = pd.DataFrame()
for key, options in column_choices.items():
    result[key] = df_merged.apply(lambda row: select_first_available(row, options), axis=1)

# Rimozione righe incomplete
df_final = result.dropna()

# Aggiunta SEQN
df_final = df_final.copy()
df_final["SEQN"] = df_merged.loc[df_final.index, "SEQN"]

# Salvataggio
df_final.to_csv(output_path, index=False)
print(f"✅ Dataset salvato in: {output_path} con {df_final.shape[0]} righe complete.")
