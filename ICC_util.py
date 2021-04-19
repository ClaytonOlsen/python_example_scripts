import pandas as pd
import pingouin as pg
import os

def icc_by_metric(df, select_metric):
    df_formatted = df[['athlete', 'id', select_metric]].pivot_table(index = ['athlete'], values = select_metric, columns = 'id')
    df_formatted = df_formatted.dropna(axis=1)
    df_formatted = df_formatted.unstack().reset_index(name='value')
    df_formatted.rename(columns={'value': select_metric},inplace =True)
    reliability = pg.intraclass_corr(data = df_formatted, targets = 'athlete', raters = 'id', ratings = select_metric).iloc[2]
    reliability_dict = {'Metric':select_metric,'Type':reliability['Type'], 'ICC':reliability['ICC'], 'CI95':reliability['CI95%']}
    return reliability_dict

def get_iccs_by_dataframe(df):
    df = df.dropna()
    df['scanned_at'] = df['scanned_at'].str.split(' ').str[0]
    df['scanned_at'] = df['scanned_at'].str.split('-')
    df['scanned_at'] = df.scanned_at.map(lambda x: int(x[0])*365 + int(x[1])*30 + int(x[2])) #turning scanned_at into a count of the approximate number of days to see the amount of time seperated
    df = df.assign(new_date = (df['scanned_at'] - df.groupby('athlete')['scanned_at'].transform(min))/30) #splits athelete trial by 30 day intervals
    df['new_date'] = df['new_date'].astype(int)

    df = df[(df.groupby(["athlete", "new_date"]).athlete.transform("size") > 1)].reset_index(drop = True) #only keeps athlete/trial date combos that have more than 1 scans. so the total amount of athletes being used to calculate ICC will be less
    df = df.assign(id= df.groupby(["athlete", 'new_date']).cumcount()) #assigns an ID for grouping athletes for the ICC calc. This will treat scans from the same athelete done at differing times as if they were different athletes.
    icc_values = pd.DataFrame(columns = ['Metric', 'Type', 'ICC', 'CI95'])
    cols_df = df.drop(columns = ['scan_sha', 'athlete', 'id', 'scanned_at', 'gender', 'new_date'])
    cols = cols_df.columns
    for item in cols:
        print(item)
        reliability = icc_by_metric(df, item)
        icc_values = icc_values.append(reliability, ignore_index=True)
    return icc_values


#Takes in folder with calc results for balance, jump, plank, or imtp. Requires string of test type, and related folder with calc resutls
def ICC_full(test_type, results_folder):
    testing_dirpath = os.getcwd() + f'/bulk_testing/results/{test_type}/{results_folder}'
    df = pd.read_csv(f'{testing_dirpath}/results_scan.csv')
    
    ICCs = get_iccs_by_dataframe(df)
    ICCs.to_csv(testing_dirpath + f'/ICC_{test_type}.csv', index = False)
    #return ICCs

k = ICC_full('balance', 'balance_eyes_open_sparta_2021-02-24')
