
import pandas as pd
import pyaging as pya
from functools import reduce
import sqlite3
import numpy as np

# === File di origine ===
data_dir = "NHANES/1988-2018/"
output_csv = "lifemina_dataset.csv"
output_sqlite = "lifemina_dataset.db"
sqlite_table = "lifemina"

#PhenoAge
#Variable		Units	Weight
#Albumin	Liver	g/L	-0.0336
#Creatinine	Kidney	umol/L	0.0095
#Glucose, serum	Metabolic	mmol/L	0.1953
#C-reactive protein (log)	Inflammation	mg/dL	0.0954
#Lymphocyte percent	Immune	%	-0.0120
#Mean cell volume	Immune	fL	0.0268
#Red cell distribution width	Immune	%	0.3306
#Alkaline phosphatase	Liver	U/L	0.0019
#White blood cell count	Immune	1000 cells/uL	0.0554
#Age		Years	0.0804

# === Specifica delle colonne per ciascun biomarcatore: (colonna, file sorgente) ===
column_choices = {
    "SEQN_new": [("SEQN_new", "response_clean.csv")],
    "albumin": [("LBDSALSI", "response_clean.csv")], #g/L
    "creatinine": [("LBXSCR", "response_clean.csv")], #mg/dL -> umol/L 
    "glucose": [("LBXSGL", "response_clean.csv")], #mg/dL -> mmol/L
    "log_crp": [("LBXCRP", "response_clean.csv")], #mg/dL
    "lymphocyte_percent": [("LBXLYPCT", "response_clean.csv")], #%
    "mean_cell_volume": [("LBXMCVSI", "response_clean.csv")], #fL
    "red_cell_distribution_width": [("LBXRDW", "response_clean.csv")], #%
    "alkaline_phosphatase": [("LBXSAPSI", "response_clean.csv")], #U/L
    "white_blood_cell_count": [("LBXWBCSI", "response_clean.csv")], #1000 cells/uL
    "age": [("RIDAGEYR", "demographics_clean.csv")],
    "DMAETHNR": [("DMAETHNR", "demographics_clean.csv")],
    "DMARACER": [("DMARACER", "demographics_clean.csv")],
    "DMDEDUC2": [("DMDEDUC2", "demographics_clean.csv")],
    "INDFMIN2": [("INDFMIN2", "demographics_clean.csv")],
    "RIAGENDR": [("RIAGENDR", "demographics_clean.csv")],
    "RIDAGEYR": [("RIDAGEYR", "demographics_clean.csv")],
    "VNAVEBPXSY": [("VNAVEBPXSY", "response_clean.csv")],
    "VNLBAVEBPXDI": [("VNLBAVEBPXDI", "response_clean.csv")],
    "BMPWHR": [("BMPWHR", "response_clean.csv")],
    "BMXBMI": [("BMXBMI", "response_clean.csv")],
    "BMXHT": [("BMXHT", "response_clean.csv")],
    "BMXWT": [("BMXWT", "response_clean.csv")],
    "HAM11": [("HAM11", "questionnaire_clean.csv")],
    "HAR1": [("HAR1", "questionnaire_clean.csv")],
    "HAR14": [("HAR14", "questionnaire_clean.csv")],
    "HAR23": [("HAR23", "questionnaire_clean.csv")],
    "HAT10": [("HAT10", "questionnaire_clean.csv")],
    "HAT12": [("HAT12", "questionnaire_clean.csv")],
    "HAT14": [("HAT14", "questionnaire_clean.csv")],
    "HAT18": [("HAT18", "questionnaire_clean.csv")],
    "HAT1S": [("HAT1S", "questionnaire_clean.csv")],
    "HAT2": [("HAT2", "questionnaire_clean.csv")],
    "HAT6": [("HAT6", "questionnaire_clean.csv")],
    "HAT8": [("HAT8", "questionnaire_clean.csv")],
    "HSD010": [("HSD010", "questionnaire_clean.csv")],
    "MORTSTAT": [("MORTSTAT", "mortality_clean.csv")],
    "PERMTH_INT": [("PERMTH_INT", "mortality_clean.csv")],
    "PERMTH_EXM": [("PERMTH_EXM", "mortality_clean.csv")],
    "VNELIGSTAT": [("VNELIGSTAT", "mortality_clean.csv")],
    "VNMORTSTAT": [("VNMORTSTAT", "mortality_clean.csv")],
    "VNUCOD_LEADING": [("VNUCOD_LEADING", "mortality_clean.csv")],
    "VNDIABETES": [("VNDIABETES", "mortality_clean.csv")],
    "VNHYPERTEN": [("VNHYPERTEN", "mortality_clean.csv")],
}

# === Filtro per file: ogni voce è (colonna, condizione_lambda) ===
filters = {
    "mortality_clean.csv": [
        ("VNELIGSTAT", lambda x: x != "Under age 18")
    ],
    "response_clean.csv": [
        ("SEQN_new", lambda x: pd.notnull(x) and str(x).startswith("I")),
        ("LBDSALSI", pd.notnull),
        ("LBXSCR", pd.notnull),
        ("LBXSGL", pd.notnull),
        ("LBXCRP", pd.notnull),
        ("LBXLYPCT", pd.notnull),
        ("LBXMCVSI", pd.notnull),
        ("LBXRDW", pd.notnull),
        ("LBXSAPSI", pd.notnull),
        ("LBXWBCSI", pd.notnull)
    ],
    "demographics_clean.csv": [
        ("RIDAGEYR", lambda x: pd.notnull(x) and x >= 18)
    ]
}

# === Funzioni di conversione per colonne specifiche ===
conversion_functions = {
    "creatinine": lambda x: x * 88.4 if pd.notnull(x) else x,     # mg/dL → µmol/L
    "glucose": lambda x: x * 0.0555 if pd.notnull(x) else x,      # mg/dL → mmol/L
    "log_crp": lambda x: np.log(x + 0.01) if pd.notnull(x) and (x + 0.01) > 0 else np.nan # CRP (mg/L) → log-scale
}

# === Caricamento file e applicazione filtri ===
files_needed = {}
for fields in column_choices.values():
    for var, file in fields:
        files_needed.setdefault(file, set()).add(var)

files_loaded = {}
for fname, vars in files_needed.items():
    df = pd.read_csv(data_dir + fname, usecols=["SEQN_new"] + list(vars), low_memory=False)
    for col, condition in filters.get(fname, []):
        if col in df.columns:
            df = df[df[col].apply(condition)]
    files_loaded[fname] = df

# === Merge progressivo su SEQN_new ===
df_merged = reduce(lambda left, right: pd.merge(left, right, on="SEQN_new", how="inner"), files_loaded.values())

# === Selezione variabili da column_choices ===
def select_first_available(row, options):
    for col, _ in options:
        if col in row and pd.notnull(row[col]):
            return row[col]
    return None

result = pd.DataFrame()
for key, options in column_choices.items():
    result[key] = df_merged.apply(lambda row: select_first_available(row, options), axis=1)

# === Applica conversioni (se definite) ===
for col, func in conversion_functions.items():
    if col in result.columns:
        result[col] = result[col].apply(func)

# === Rimuove righe incomplete
df_final = result#.dropna().copy()

# Colonne richieste
phenoage_columns = [
    "albumin", "creatinine", "glucose", "log_crp", "lymphocyte_percent",
    "mean_cell_volume", "red_cell_distribution_width",
    "alkaline_phosphatase", "white_blood_cell_count", "age"
]

# Assicura che le colonne esistano
missing = [col for col in phenoage_columns if col not in df_final.columns]
if missing:
    raise ValueError(f"❌ Colonne mancanti per PhenoAge: {missing}")

# Filtra solo le righe complete
df_pheno = df_final[phenoage_columns].dropna()
if df_pheno.empty:
    raise ValueError("❌ Nessuna riga completa per calcolare PhenoAge")

# Converte in formato AnnData
adata = pya.preprocess.df_to_adata(df_pheno, verbose=True)

# Calcola PhenoAge
pya.pred.predict_age(adata, ["PhenoAge"], verbose=True)

col_name = next((c for c in adata.obs.columns if c.lower() == "phenoage"), None)
if col_name is None:
    raise ValueError("❌ 'PhenoAge' non trovato in adata.obs")

df_final.loc[df_pheno.index, "PhenoAge"] = adata.obs[col_name].values

df_final.to_csv(output_csv, index=False)
print(f"✅ Dataset CSV salvato in: {output_csv} ({df_final.shape[0]} righe)")

# === Salvataggio su SQLite
conn = sqlite3.connect(output_sqlite)
df_final.to_sql(sqlite_table, conn, if_exists="replace", index=False)
conn.close()
print(f"✅ Dataset salvato anche su SQLite: {output_sqlite} (tabella '{sqlite_table}')")