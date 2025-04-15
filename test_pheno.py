import pandas as pd
import pyaging as pya
import kagglehub
from kagglehub import KaggleDatasetAdapter
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]


# Set the path to the file you'd like to load
file_path = "./HNANES/1988-2018"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "nguyenvy/nhanes-19882018",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())

#pya.data.download_example_data('blood_chemistry_example', verbose=False)
#df = pd.read_pickle('pyaging_data/blood_chemistry_example.pkl')
df = pd.read_csv('output.csv') 
df.index = df["Unnamed: 0"]
df = df.drop(columns=["Unnamed: 0"])

pd.set_option('display.max_columns', None)
print(df.dtypes)

adata = pya.preprocess.df_to_adata(df, verbose=False)
pya.pred.predict_age(adata, ['PhenoAge'], verbose=False)

print(adata.obs)