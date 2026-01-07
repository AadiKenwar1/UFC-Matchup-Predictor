import pandas as pd

# Parse METHOD to categorize finish type
def categorize_method(method):
    if pd.isna(method) or method == '--':
        return None
    method_str = str(method).lower().strip()
    if 'ko' in method_str or 'tko' in method_str:
        return 'KO/TKO'
    elif 'submission' in method_str or 'sub' in method_str:
        return 'Submission'
    elif 'decision' in method_str:
        return 'Decision'
    return None

# Helper function to calculate historical averages
def calc_historical_avg(df, fighter_num, stat_name, window=3):
    col_name = f'fighter{fighter_num}_{stat_name}'
    avg_name = f'fighter{fighter_num}_avg_{stat_name}_last_{window}'
    avg = df.groupby(f'fighter{fighter_num}_name')[col_name].apply(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    df[avg_name] = avg.values
    df[avg_name] = df[avg_name].fillna(df[avg_name].mean())
    return df

