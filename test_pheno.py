import pandas as pd
import pyaging as pya
pya.data.download_example_data('blood_chemistry_example', verbose=False)
df = pd.read_pickle('pyaging_data/blood_chemistry_example.pkl')
#df = pd.read_csv('phenoage_tuples_final.csv', low_memory=False) 
#df.index = df["Unnamed: 0"]
#df = df.drop(columns=["Unnamed: 0"])

pd.set_option('display.max_columns', None)
print(df.dtypes)

adata = pya.preprocess.df_to_adata(df, verbose=False)
pya.pred.predict_age(adata, ['PhenoAge'], verbose=False)

print(adata.obs)