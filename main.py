import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

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

#EDA
#creating new features for EDA for further insides (partially feature engineering)

logging.info('Profitability and CF - metrics ratios...')
pivot_df['FCF_margin'] = pivot_df['Free Cash Flow'] / pivot_df['End Cash'] # FCF / End Cash = FCFmargin
pivot_df['OCF_margin'] = pivot_df['Operating Cash Flow'] / pivot_df['End Cash'] # OCF / ENd cash = OCF margin

logging.info('Growth and Change ratios...')
pivot_df['Net_Income_YoY'] = pivot_df.groupby('company')['Net Income'].pct_change() * 100
print(f'YoY trend analysis of Net Income: ', pivot_df['Net_Income_YoY'])

pivot_df['FCF_YoY'] = pivot_df.groupby('company')['Free Cash Flow'].pct_change() * 100
print(f'YoY trend analysis of FCF: ', pivot_df['FCF_YoY'])

pivot_df['OCF_YoY'] = pivot_df.groupby('company')['Operating Cash Flow'].pct_change() * 100
print(f'YoY trend analysis of OCF: ', pivot_df['OCF_YoY'])

#now comparing Cap Spending of each company

pivot_df['capex_ratio'] = pivot_df['Capital Expenditures'] / pivot_df['End Cash'] # Capex ratio = capex / end cash
pivot_df['debt_repay_ratio'] = pivot_df['Debt Repay.'] / pivot_df['End Cash'] # debt repay ration = debt repay / end cash

#EDA visuals
sns.set_style('whitegrid')

plt.figure(figsize=(8, 6))
sns.countplot(data=pivot_df, x='burn_cash', palette=['green', 'red'])
plt.xticks([0,1], ['No burn', 'Burn'])
plt.xlabel('Burn Cash 1/0')
plt.title('Distribution of class')
plt.tight_layout()
plt.show() # this will show balance 

burn_rate = pivot_df.groupby('company')['burn_cash'].mean().reset_index()
burn_rate['burn_cash_percent'] = burn_rate['burn_cash'] * 100

plt.figure(figsize=(10, 8))
sns.barplot(data=burn_rate, x='company', y='burn_cash_percent', color='skyblue')
plt.xlabel('Company')
plt.ylabel('Burn cash percentage')
plt.title('Percentage of years with cash burn')
plt.tight_layout()
plt.show() #this will show burn rate 

avg_margins = pivot_df.groupby('company')[['FCF_margin', 'OCF_margin']].mean().reset_index() #
avg_margins = avg_margins.melt(
    id_vars='company',
    var_name='Margin Type',
    value_name='Value'
) #need to convert from wide to long format

plt.figure(figsize=(10, 6))
sns.barplot(data=avg_margins, x='company', y='Value', hue='Margin Type')
plt.axhline(0, color='black', linewidth=1)
plt.title('Avg margins OCF and FCF')
plt.tight_layout()
plt.show() # avg margins for each company - FCF and OCF

trend = pivot_df.groupby('year')['Free Cash Flow'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=trend, x='year', y='Free Cash Flow', marker='o')
plt.title('Trend')
plt.tight_layout()
plt.show() # FCF trends

capex_ratio = pivot_df.groupby('company')['capex_ratio'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=capex_ratio, x='company', y='capex_ratio', color='skyblue')
plt.ylabel('Capex / End cash')
plt.title('avg capex ratio ')
plt.tight_layout()
plt.show() # shows avg capex ratio by company





