import warnings
import h2o
from h2o import H2OFrame
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

h2o.init(max_mem_size='4G')
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

logging.info('Importing all the necessary .cvs files (10-K reports)...')
#importing .cvs file for all companies within one sector

df_cej = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/CEG_cleaned.csv')
df_ccj = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/CCJ_cleaned.csv')
df_bwxt = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/BWXT_cleaned.csv')
df_gev = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/GEV_cleaned.csv')
df_leu = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/LEU_cleaned.csv')

logging.info('Combining .csv files...')
# adding 'company' column to each dataframe
df_cej['company'] = 'CEG'
df_ccj['company'] = 'CCJ'
df_bwxt['company'] = 'BWXT'
df_gev['company'] = 'GEV'
df_leu['company'] = 'LEU'

combined_df = pd.concat([df_cej, df_ccj, df_bwxt, df_gev, df_leu], ignore_index=True)
combined_df.to_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/combined_10k_reports.csv', index=False)

logging.info('I am trying to covert wide -> long format...')
#these below four don't contain any information as they are HEADER ROWS
combined_df = combined_df[~combined_df['Unnamed: 0'].isin(['Operating Activities', 'Investing Activities', 'Financing Activities', 'Supplemental Information'])] 

#melting to long format
long_df = pd.melt(
    combined_df,
    id_vars=['Unnamed: 0', 'company'], # columns to keep as identifiers
    value_vars=['2021', '2022', '2023', '2024'], # columns to melt into rows (they were columns) so I get one row per metric per company per year
    var_name='year', # name for newly transformed feature with years
    value_name='value' # new column 'value'
)

# to prevent errors, converting into numerical values
long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce')

#pivoting to create columns for each metric
pivot_df = long_df.pivot_table(
    index=['company', 'year'], # each column -> 1 year, one company
    columns=['Unnamed: 0'], # each metric - separate column
    values='value', #use financial data
).reset_index()

#making sure all values are numeric
numeric_col = pivot_df.columns[2:] # to skip 'company' and 'year'
for c in numeric_col:
    pivot_df[c] = pd.to_numeric(pivot_df[c], errors='coerce' ) #'coerce' prevents from errors

#for classification goal, creating a target variable
pivot_df = pivot_df.sort_values(['company', 'year']) #important chronological order as this is financial data
pivot_df['FCF_next_year'] = pivot_df.groupby('company')['Free Cash Flow'].shift(-1) #next year cash flow column
pivot_df['OCF_next_year'] = pivot_df.groupby('company')['Operating Cash Flow'].shift(-1) #next year operating cash flow column
pivot_df['burn_cash'] = (
    (pivot_df['FCF_next_year'] < 0) | (pivot_df['OCF_next_year'] < 0) # | = or; astype(int) converts boolean to binary 1/0
).astype(int)

pivot_df = pivot_df.dropna(subset=['burn_cash']) # removes rows where there is no target (important for errors)
pivot_df.to_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/all_companies_with_target.csv', index=False) # saving combined pivoted dataset 

for c in pivot_df['company'].unique():
    company_pivot = pivot_df[pivot_df['company'] == c]
    company_pivot.to_csv(f'{c}_with_target.csv', index=False)


#STAGE 1: DATA INSPECTION
logging.info('Starting data inspection...')
logging.info('Checking shape and data types...')
print(pivot_df.shape)
print(pivot_df.dtypes)

logging.info('First 10 rows of data')
print(pivot_df.head(10))

logging.info('List of all features...')
print(pivot_df.columns)

logging.info('Value counts...')
print(pivot_df['burn_cash'].value_counts())




