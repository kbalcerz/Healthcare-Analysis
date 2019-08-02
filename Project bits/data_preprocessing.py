import pandas as pd
import numpy as np

df = pd.read_excel('Nursing_2014.xlsx', sheet_name = 'Provider')

df_len = len(df)

drop_col_list = []


def handleNull(col):
    if(sum(pd.isna(df[col])) < df_len/2):
        if(df[col].dtype != 'object'):
            return 1
        if(len(df[df[col] == '*']) < df_len/2):
            temp = df[col].copy()
            temp[temp == '*'] = 0
            df[col][df[col] == '*'] = temp.mean()
            return 1
        else:
            return 0        
    else:
        return 0           


for col in df.columns[6:]:
    val = handleNull(col)
    if(val == 0):
        drop_col_list.append(col)    

df = df.drop(columns = drop_col_list)

output_file = 'Nursing_2014_processed.csv'
df.to_csv(output_file, index = False)


