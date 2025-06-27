import logging
import pandas as pd


logging.info('Importing all the necessary .cvs files (10-K reports)...')
#importing .cvs file for all companies within one sector

df_cej = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/CEG_cleaned.csv')
df_ccj = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/CCJ_cleaned.csv')
df_bwxt = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/BWXT_cleaned.csv')
df_gev = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/GEV_cleaned.csv')
df_leu = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/LEU_cleaned.csv')
df_anldf = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/ANLDF_cleaned.csv')
df_boe = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/BOE_cleaned.csv')
df_nee = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/NEE_cleaned.csv')
df_nne = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/NNE_cleaned.csv')
df_oklo = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/OKLO_cleaned.csv')
df_uec = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/UEC_cleaned.csv')
df_nxe = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/NXE_cleaned.csv')
df_uuuu = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/UUUU_cleaned.csv')


df_aes = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/AES_cleaned.csv')
df_boeax = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/BOE.AX_cleaned.csv')
df_cpxto = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/CPX.TO_cleaned.csv')
df_dte = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/DTE_cleaned.csv')
df_cwen = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/CWEN_cleaned.csv')
df_flnc = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/FLNC_cleaned.csv')
df_enph = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/ENPH_cleaned.csv')
df_fslr = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/FSLR_cleaned.csv')
df_sedg = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/SEDG_cleaned.csv')
df_smr = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/SMR_cleaned.csv')
df_so = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/SO_cleaned.csv')
df_ssel = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/SSE.L_cleaned.csv')
df_ssezy = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/SSEZY_cleaned.csv')
df_tsla = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/TSLA_cleaned.csv')
df_vst = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/VST_cleaned.csv')
df_xel = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/XEL_cleaned.csv')

logging.info('Combining .csv files...')
# adding 'company' column to each dataframe
df_cej['company'] = 'CEG'
df_ccj['company'] = 'CCJ'
df_bwxt['company'] = 'BWXT'
df_gev['company'] = 'GEV'
df_leu['company'] = 'LEU'
df_anldf['company'] = 'ANLDF'
df_boe['company'] = 'BOE'
df_nee['company'] = 'NEE'
df_nne['company'] = 'NNE'
df_oklo['company'] = 'OKLO'
df_uec['company'] = 'UEC'
df_nxe['company'] = 'NXE'
df_uuuu['company'] = 'UUUU'
df_aes['company'] = 'AES'
df_boeax['company'] = 'BOE.AX'
df_cpxto['company'] = 'CPX.TO'
df_dte['company'] = 'DTE'
df_cwen['company'] = 'CWEN'
df_flnc['company'] = 'FLNC'
df_enph['company'] = 'ENPH'
df_fslr['company'] = 'FSLR'
df_sedg['company'] = 'SEDG'
df_smr['company'] = 'SMR'
df_so['company'] = 'SO'
df_ssel['company'] = 'SSE.L'
df_ssezy['company'] = 'SSEZY'
df_tsla['company'] = 'TSLA'
df_vst['company'] = 'VST'
df_xel['company'] = 'XEL'



combined_df = pd.concat([df_cej, df_ccj, df_bwxt, df_gev, df_leu, df_anldf, df_boe, df_nee, df_nxe, df_nne, df_oklo, df_uec, df_uuuu, df_aes, df_boeax, df_cpxto,
                        df_dte, df_cwen, df_flnc, df_enph, df_fslr, df_sedg, df_smr, df_so, df_ssel, df_tsla, df_vst, df_xel], ignore_index=True) 
combined_df.to_csv('/Users/ohavryleshko/Documents/GitHub/AutoML/10k_reports/csv_ready/combined_10k_reports.csv', index=False) #combining all files into one dataset
