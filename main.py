import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

logging.info('Importing combined .csv...')
combined_df = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/combined_10k.csv')

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


pivot_df['year'] = pd.to_numeric(pivot_df['year'], errors='coerce')
pivot_df = pivot_df.dropna(subset=['year'])
pivot_df['year'] = pivot_df['year'].astype(int)

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
logging.info('Finished with data inspection.')

#EDA
#creating new features for EDA for further insides (partially feature engineering)
logging.info('Starting EDA and Feature Engineering...')
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
plt.xticks(ha='right', rotation='vertical')
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
plt.xticks(ha='right', rotation='vertical')
plt.axhline(0, color='black', linewidth=1)
plt.title('Avg margins OCF and FCF')
plt.tight_layout()
plt.show() # avg margins for each company - FCF and OCF

trend = pivot_df.groupby('year')['Free Cash Flow'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=trend, x='year', y='Free Cash Flow', marker='o')
plt.xticks(ha='right', rotation='vertical')
plt.title('Trend')
plt.tight_layout()
plt.show() # FCF trends

capex_ratio = pivot_df.groupby('company')['capex_ratio'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=capex_ratio, x='company', y='capex_ratio', color='skyblue')
plt.xticks(ha='right', rotation='vertical')
plt.ylabel('Capex / End cash')
plt.title('avg capex ratio ')
plt.tight_layout()
plt.show() # shows avg capex ratio by company

logging.info('Finished EDA and Feature Engineering.')

# modeling stage (PyCaret)

logging.info('Starting with modeling using PyCaret...')

from pycaret.classification import setup, compare_models, evaluate_model, predict_model, tune_model, interpret_model, save_model, load_model

df = pivot_df.drop(['company', 'year'], axis=1)
features = pivot_df.columns
trg = 'burn_cash'

df = df.replace([float('inf'), float('-inf')], pd.NA)
df = df.dropna()

clf = setup(data=df, target=trg, session_id=42, normalize=True, fix_imbalance=True) #initialising the setup
best_model = compare_models() # return best-performing model
tuned_model = tune_model(best_model)
evaluate_model(tuned_model) # getting evaluation of the best_model

predictions = predict_model(tuned_model)
print('Predictions: ', predictions)

interpret_model(tuned_model) # SHAP explanation, which features influence 'burn_cash' prediction the most
save_model(tuned_model, 'burn_cash_model') 

model = load_model('burn_cash_model')

logging.info('FINISHED MODELING WITH PYCARET.')

#visualising the resuls
from pycaret.classification import plot_model
plot_model(best_model, plot='feature') # plot for feature importance - what drives the prediction of cash burn 
plot_model(best_model, plot='confusion_matrix') # confusion matrixx
plot_model(best_model, plot='auc')
plot_model(best_model, plot='pr')

# Modeling stage with H2o

# import h2o
# from h2o import H2OFrame
# from h2o.automl import H2OAutoML

# h2o.init()

# dataframe = h2o.H2OFrame(pivot_df)
# dataframe['burn_cash'] = dataframe['burn_cash'].asfactor()

# target = 'burn_cash'
# features = [c for c in dataframe.columns if c not in ['burn_cash', 'year', 'company']]

# train, valid, test = dataframe.split_frame(ratios=[0.8, 0.1], seed=42)

# aml = H2OAutoML(
#     max_models=10,
#     max_runtime_secs=450,
#     keep_cross_validation_predictions=True,
#     nfolds=3)

# print('Train columns: ', train.columns)
# print("Feature columns (x):", features)

# aml.train(x=features, y=target, training_frame=train, validation_frame=valid)
# logging.info('Creating leaderboard and printing out top 5 models...')
# lbd = aml.leaderboard # leaderboard of top 5 models
# print(f'Top 5 best-performing models: ', lbd.head(5))

# logging.info('Finding the best model among leaders...')
# best_model = aml.leader

# predictions = best_model.predict(test)
# performance = best_model.model_performance(test_data=test)
# performance.confusion_matrix()

# h2o.save_model(best_model, path='/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/models', force=True) # force overrides the file without asking