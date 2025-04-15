
import pandas as pd

# === Percorsi file ===
resp_path = "NHANES/1988-2018/response_clean.csv"
demo_path = "NHANES/1988-2018/demographics_clean.csv"
output_path = "phenoage_tuples_final.csv"

# === Colonne selezionate per ciascun biomarcatore ===
column_choices = {
    "albumin": ["LBXSAL"],
    "creatinine": ["LBXSCR"],
    "glucose": ["LBXSGL"],
    "log_crp": ["LBXHSCRP"],
    "lymphocyte_percent": ["LBXLYPCT"],
    "mean_cell_volume": ["LBXMCVSI"],
    "red_cell_distribution_width": ["LBXRDW"],
    "alkaline_phosphatase": ["LBXSAPSI"],
    "white_blood_cell_count": ["LBXWBCSI"],
    "age": ["RIDAGEYR"]
}

# === Carica i dataset ===
df_resp = pd.read_csv(resp_path, low_memory=False)
df_demo = pd.read_csv(demo_path, low_memory=False)

# Merge dei dati su SEQN
df = df_resp.merge(df_demo[["SEQN"] + column_choices["age"]], on="SEQN", how="inner")

# Funzione per selezionare la prima colonna valida
def select_first_available(row, options):
    for col in options:
        if col in row and pd.notna(row[col]):
            return row[col]
    return None

# Costruzione delle colonne finali
result = pd.DataFrame()
for key, options in column_choices.items():
    result[key] = df.apply(lambda row: select_first_available(row, options), axis=1)

# Rimuove le righe con valori mancanti
df_final = result.dropna()

# Aggiunge SEQN per riferimento
df_final["SEQN"] = df.loc[df_final.index, "SEQN"]

# Salvataggio del file
df_final.to_csv(output_path, index=False)
print(f"✅ Tupla PhenoAge salvata in: {output_path} con {df_final.shape[0]} righe complete.")
