import pandas as pd

def clean_10k_file(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath) # i mention filepath when calling the function
    for col in df.select_dtypes(include='object').columns:
        # remove $, commas, replace parentheses for negatives
        df[col] = (
            df[col] # below - replacting all the symbols
            .str.replace('$', '', regex=False) 
            .str.replace(',', '', regex=False)
            .str.replace('(', '-', regex=False) 
            .str.replace(')', '', regex=False)
            .str.strip()  # remove whitespace if any
        )
        # convert to numeric if possible
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    return df # returning each data frame (.csv)

df_clean = clean_10k_file('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_raw/LEU_ANNUAL_CASH_FLOW_FROM_PERPLEXITY.csv') # just change ticker of the company
df_clean.to_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/LEU_cleaned.csv', index=False) # change the ticker here as well otherwise file won't be saved

