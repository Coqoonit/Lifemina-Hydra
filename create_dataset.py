
import pandas as pd
from scipy.stats.mstats import zscore
import pyaging as pya
from functools import reduce
import sqlite3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pycox.models import CoxPH, CoxTime
from torchtuples import Model
import torchtuples as tt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pycox.evaluation import EvalSurv

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
    "ETHNICITY": [("DMAETHNR", "demographics_clean.csv")],
    "RACE": [("DMARACER", "demographics_clean.csv")],
    "EDUCATION": [("DMDEDUC2", "demographics_clean.csv")],
    "INCOME": [("INDFMIN2", "demographics_clean.csv")],
    "GENDER": [("RIAGENDR", "demographics_clean.csv")],
    "SYSTOLIC_PRESSURE": [("VNAVEBPXSY", "response_clean.csv")],
    "DIASTOLIC_PRESSURE": [("VNLBAVEBPXDI", "response_clean.csv")],
    "WAIST_HIP_RATIO": [("BMPWHR", "response_clean.csv")],
    "BMI": [("BMXBMI", "response_clean.csv")],
    "HEIGHT": [("BMXHT", "response_clean.csv")],
    "WEIGHT": [("BMXWT", "response_clean.csv")],
    "SMOKER": [("HAR1", "questionnaire_clean.csv")],
    "TABACCO": [("HAR14", "questionnaire_clean.csv")],
    "CIGARS": [("HAR23", "questionnaire_clean.csv")],
    "DANCE_LAST_MONTH": [("HAT10", "questionnaire_clean.csv")],
    "EXERCISE_LAST_MONTH": [("HAT12", "questionnaire_clean.csv")],
    "GARDENER_LAST_MONTH": [("HAT14", "questionnaire_clean.csv")],
    "SPORT_LAST_MONTH": [("HAT18", "questionnaire_clean.csv")],
    "WALK_LAST_MONTH": [("HAT1S", "questionnaire_clean.csv")],
    "RUN_LAST_MONTH": [("HAT2", "questionnaire_clean.csv")],
    "SWIM_LAST_MONTH": [("HAT6", "questionnaire_clean.csv")],
    "AEROBICS_LAST_MONTH": [("HAT8", "questionnaire_clean.csv")],
    "GENERAL_HEALTH_CONDITION": [("HSD010", "questionnaire_clean.csv")],
    "MORTSTAT": [("MORTSTAT", "mortality_clean.csv")],
    "DEATH_MONTHS": [("PERMTH_INT", "mortality_clean.csv")],
    "DEATH_REASON": [("VNUCOD_LEADING", "mortality_clean.csv")],
    "DIABETES": [("VNDIABETES", "mortality_clean.csv")],
    "HYPERTENSION": [("VNHYPERTEN", "mortality_clean.csv")],
}

# === Filtro per file: ogni voce è (colonna, condizione_lambda) ===
filters = {
    "mortality_clean.csv": [
        ("VNELIGSTAT", lambda x: x != "Under age 18"),
        ("VNUCOD_LEADING", lambda x: x != "All other causes (residual)" and x != "Accidents (unintentional injuries) (112-123)")
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

df_final['DIABETES'] = df_final['DIABETES'].map({
    'Yes': 1,
    'No': 0
})

df_final['HYPERTENSION'] = df_final['HYPERTENSION'].map({
    'Yes': 1,
    'No': 0
})

# Colonne da normalizzare per Cox
columns_to_normalize = [
    'PhenoAge', 'age', 'SYSTOLIC_PRESSURE', 'DIASTOLIC_PRESSURE',
    'WAIST_HIP_RATIO', 'BMI', 'HEIGHT', 'WEIGHT',
    'albumin','creatinine','glucose','log_crp','lymphocyte_percent','mean_cell_volume',
    'red_cell_distribution_width','alkaline_phosphatase','white_blood_cell_count'
]

# Filtra solo le colonne esistenti
columns_to_normalize = [col for col in columns_to_normalize if col in df_final.columns]

# Sostituisci Inf e -Inf con NaN
df_final[columns_to_normalize] = df_final[columns_to_normalize].replace([np.inf, -np.inf], np.nan)

# Converti esplicitamente le colonne in float64
df_final[columns_to_normalize] = df_final[columns_to_normalize].astype("float64")

# Applica Z-score ignorando i NaN
df_final.loc[:, columns_to_normalize] = df_final[columns_to_normalize].apply(
    lambda x: zscore(x, nan_policy='omit')
)

# === Salvataggio del dataset finale ===
df_final.to_csv(output_csv, index=False)
print(f"✅ Dataset CSV salvato in: {output_csv} ({df_final.shape[0]} righe)")

# === Salvataggio su SQLite
conn = sqlite3.connect(output_sqlite)
df_final.to_sql(sqlite_table, conn, if_exists="replace", index=False)

print(f"✅ Dataset salvato anche su SQLite: {output_sqlite} (tabella '{sqlite_table}')")

# === Visualizzazione delle correlazioni ===
'''
correlation_matrix = df_final.corr(numeric_only=True)

subset_corr =  correlation_matrix.loc[['age', 'PhenoAge', 'MORTSTAT'], :]
plt.figure(figsize=(12, 10))
sns.heatmap(
    subset_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    annot_kws={"size": 8}  # 👈 Riduce la dimensione del testo sulle celle
)
plt.title("Correlation Matrix", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=8)  # 👈 Etichette asse X
plt.yticks(fontsize=8)                           # 👈 Etichette asse Y
plt.tight_layout()
plt.show()
'''


# 🔹 Definizione delle colonne per pycox
columns_features = [
    'PhenoAge', 'age', 'SYSTOLIC_PRESSURE', 'DIASTOLIC_PRESSURE',
    'WAIST_HIP_RATIO', 'BMI', 'HEIGHT', 'WEIGHT',
    'albumin', 'creatinine', 'glucose', 'log_crp', 'lymphocyte_percent',
    'mean_cell_volume', 'red_cell_distribution_width',
    'alkaline_phosphatase', 'white_blood_cell_count'
]

# 🔹 Prepara dataset per pycox
df_surv = df_final.dropna(subset=['DEATH_MONTHS', 'MORTSTAT'] + columns_features)
df_surv = df_surv[df_surv["DEATH_MONTHS"] >= 0]
X = df_surv[columns_features].values.astype('float32')
y = df_surv[['DEATH_MONTHS', 'MORTSTAT']].values.astype('float32')

# 🔹 Normalizzazione
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔹 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Model definition
net = torch.nn.Sequential(
    torch.nn.Linear(X.shape[1], 32),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(32),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(32, 1)
)

model = CoxPH(net, tt.optim.Adam)
model.fit(X_train, (y_train[:,0], y_train[:,1]), batch_size=256, epochs=100, verbose=True)
model.compute_baseline_hazards()

# 🔹 Evaluation
surv = model.predict_surv_df(X_test)
ev = EvalSurv(surv, y_test[:,0], y_test[:,1], censor_surv='km')
print("Concordance index (C-index):", ev.concordance_td())

# 🔍 Analisi: Concordance Index per singola variabile
print("\n🔎 Analisi individuale delle feature (C-index univariato):")
from collections import OrderedDict

scores = {}

for col in columns_features:
    print(f"🎯 Variabile: {col}")
    X_single = df_surv[[col]].values.astype("float32")
    scaler = StandardScaler()
    X_single = scaler.fit_transform(X_single)

    X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)

    net = torch.nn.Sequential(torch.nn.Linear(1, 1))
    model = CoxPH(net, tt.optim.Adam)
    model.fit(X_train, (y_train[:, 0], y_train[:, 1]), batch_size=256, epochs=100, verbose=False)
    model.compute_baseline_hazards()

    surv = model.predict_surv_df(X_test)
    ev = EvalSurv(surv, y_test[:, 0], y_test[:, 1], censor_surv="km")
    scores[col] = ev.concordance_td()

# Ordina e stampa
scores_sorted = OrderedDict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
print("\n📊 C-index per singola variabile:")
for k, v in scores_sorted.items():
    print(f"{k:<30} {v:.4f}")
