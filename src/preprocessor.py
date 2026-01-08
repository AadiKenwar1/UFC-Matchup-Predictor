import pandas as pd
import numpy as np
from pathlib import Path

# Get project root directory (go up from src/preprocessor.py)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Parse fraction strings to (landed, attempted) tuple, e.g. '17 of 26'
def parse_fraction(value):
    if pd.isna(value) or value in ['---', '--']:
        return (0, 0)
    parts = str(value).split(' of ')
    return (int(parts[0]), int(parts[1])) if len(parts) == 2 else (0, 0)

# Parse percentage strings to decimal values, e.g. '65%' to 0.65
def parse_percentage(value):
    if pd.isna(value) or value in ['---', '--']:
        return 0.0
    return float(str(value).replace('%', '')) / 100

# Parse time strings to total seconds, e.g. '0:04' to 4
def parse_time_seconds(value):
    if pd.isna(value) or value in ['---', '--', '0:00']:
        return 0
    parts = str(value).split(':')
    return int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else 0

# Parse height strings to total inches, e.g. "5' 9\"" to 69
def parse_height_inches(value):
    if pd.isna(value) or value == '--':
        return np.nan
    parts = str(value).replace('"', '').split("'")
    if len(parts) == 2:
        return int(parts[0].strip()) * 12 + int(parts[1].strip())
    return np.nan

# Parse weight strings to numeric pounds, e.g. '125 lbs.' to 125
def parse_weight_lbs(value):
    if pd.isna(value) or value == '--':
        return np.nan
    return int(str(value).replace(' lbs.', ''))

# Parse reach strings to numeric inches, e.g. '68"' to 68
def parse_reach_inches(value):
    if pd.isna(value) or value == '--':
        return np.nan
    return int(str(value).replace('"', ''))

# Parse DOB strings to datetime, handling '--' as NaT
def parse_dob(value):
    if pd.isna(value) or value == '--':
        return pd.NaT
    return pd.to_datetime(value, errors='coerce')

# Combine all UFC CSVs into a single dataset for ML prediction
def combine_dataframes():
    # Load CSVs using absolute paths
    events = pd.read_csv(DATA_DIR / 'ufc_event_details.csv')
    results = pd.read_csv(DATA_DIR / 'ufc_fight_results.csv')
    stats = pd.read_csv(DATA_DIR / 'ufc_fight_stats.csv')
    fighter_tott = pd.read_csv(DATA_DIR / 'ufc_fighter_tott.csv')
    # Strip whitespace from EVENT and BOUT columns to fix merge issues
    events['EVENT'] = events['EVENT'].str.strip()
    results['EVENT'] = results['EVENT'].str.strip()
    results['BOUT'] = results['BOUT'].str.strip()
    stats['EVENT'] = stats['EVENT'].str.strip()
    stats['BOUT'] = stats['BOUT'].str.strip()
    # Start with results (has target variable)
    df = results.copy()
    # Add event metadata
    df = df.merge(events[['EVENT', 'DATE', 'LOCATION']], on='EVENT', how='left')
    # Convert DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    # Extract fighter names
    df[['fighter1_name', 'fighter2_name']] = df['BOUT'].str.split(' vs. ', expand=True)
    df['fighter1_name'] = df['fighter1_name'].str.strip()
    df['fighter2_name'] = df['fighter2_name'].str.strip()
    # Parse and aggregate fight stats per fighter
    stats_clean = stats.copy()
    stats_clean['sig_strikes_landed'] = stats_clean['SIG.STR.'].apply(lambda x: parse_fraction(x)[0])
    stats_clean['sig_strikes_attempted'] = stats_clean['SIG.STR.'].apply(lambda x: parse_fraction(x)[1])
    stats_clean['sig_strikes_pct'] = stats_clean['SIG.STR. %'].apply(parse_percentage)
    stats_clean['total_strikes_landed'] = stats_clean['TOTAL STR.'].apply(lambda x: parse_fraction(x)[0])
    stats_clean['total_strikes_attempted'] = stats_clean['TOTAL STR.'].apply(lambda x: parse_fraction(x)[1])
    stats_clean['takedowns_landed'] = stats_clean['TD'].apply(lambda x: parse_fraction(x)[0])
    stats_clean['takedowns_attempted'] = stats_clean['TD'].apply(lambda x: parse_fraction(x)[1])
    stats_clean['control_time_sec'] = stats_clean['CTRL'].apply(parse_time_seconds)
    stats_clean['head_landed'] = stats_clean['HEAD'].apply(lambda x: parse_fraction(x)[0])
    stats_clean['body_landed'] = stats_clean['BODY'].apply(lambda x: parse_fraction(x)[0])
    stats_clean['leg_landed'] = stats_clean['LEG'].apply(lambda x: parse_fraction(x)[0])
    stats_clean['distance_landed'] = stats_clean['DISTANCE'].apply(lambda x: parse_fraction(x)[0])
    stats_clean['clinch_landed'] = stats_clean['CLINCH'].apply(lambda x: parse_fraction(x)[0])
    stats_clean['ground_landed'] = stats_clean['GROUND'].apply(lambda x: parse_fraction(x)[0])
    stats_clean['takedown_pct'] = stats_clean['TD %'].apply(parse_percentage)
    # Aggregate per fighter per fight
    agg_dict = {
        'sig_strikes_landed': 'sum', 'sig_strikes_attempted': 'sum',
        'sig_strikes_pct': 'mean', 'total_strikes_landed': 'sum',
        'total_strikes_attempted': 'sum', 'takedowns_landed': 'sum',
        'takedowns_attempted': 'sum', 'takedown_pct': 'mean',
        'control_time_sec': 'sum', 'head_landed': 'sum', 'body_landed': 'sum',
        'leg_landed': 'sum', 'distance_landed': 'sum', 'clinch_landed': 'sum',
        'ground_landed': 'sum', 'KD': 'sum', 'SUB.ATT': 'sum', 'REV.': 'sum'
    }
    stats_agg = stats_clean.groupby(['EVENT', 'BOUT', 'FIGHTER']).agg(agg_dict).reset_index()
    # Merge fighter1 stats
    stats_f1 = stats_agg.rename(columns={col: f'fighter1_{col}' for col in agg_dict.keys()})
    df = df.merge(stats_f1, left_on=['EVENT', 'BOUT', 'fighter1_name'], 
                  right_on=['EVENT', 'BOUT', 'FIGHTER'], how='left')
    df = df.drop(columns=['FIGHTER'])
    # Merge fighter2 stats
    stats_f2 = stats_agg.rename(columns={col: f'fighter2_{col}' for col in agg_dict.keys()})
    df = df.merge(stats_f2, left_on=['EVENT', 'BOUT', 'fighter2_name'],
                  right_on=['EVENT', 'BOUT', 'FIGHTER'], how='left')
    df = df.drop(columns=['FIGHTER'])
    # Add fighter attributes
    fighter_tott['height_inches'] = fighter_tott['HEIGHT'].apply(parse_height_inches)
    fighter_tott['weight_lbs'] = fighter_tott['WEIGHT'].apply(parse_weight_lbs)
    fighter_tott['reach_inches'] = fighter_tott['REACH'].apply(parse_reach_inches)
    fighter_tott['dob_datetime'] = fighter_tott['DOB'].apply(parse_dob)
    # Merge fighter1 attributes
    df = df.merge(fighter_tott[['FIGHTER', 'height_inches', 'weight_lbs', 'reach_inches', 'STANCE', 'dob_datetime']],
                  left_on='fighter1_name', right_on='FIGHTER', how='left')
    df = df.rename(columns={'height_inches': 'fighter1_height', 'weight_lbs': 'fighter1_weight',
                           'reach_inches': 'fighter1_reach', 'STANCE': 'fighter1_stance', 'dob_datetime': 'fighter1_dob'})
    df = df.drop(columns=['FIGHTER'])
    # Merge fighter2 attributes
    df = df.merge(fighter_tott[['FIGHTER', 'height_inches', 'weight_lbs', 'reach_inches', 'STANCE', 'dob_datetime']],
                  left_on='fighter2_name', right_on='FIGHTER', how='left')
    df = df.rename(columns={'height_inches': 'fighter2_height', 'weight_lbs': 'fighter2_weight',
                           'reach_inches': 'fighter2_reach', 'STANCE': 'fighter2_stance', 'dob_datetime': 'fighter2_dob'})
    df = df.drop(columns=['FIGHTER'])
    
    return df

# Get mode value from a series, returning NaN if no mode exists
def get_mode(series):
    mode_values = series.mode()
    return mode_values.iloc[0] if len(mode_values) > 0 else np.nan

# Fill NaN values using weight-class-specific means
def fill_nan_values(df):
    # Fill NaN values for height, weight, reach using weight-class-specific means
    for attr in ['height', 'weight', 'reach']:
        for fighter_num in [1, 2]:
            col = f'fighter{fighter_num}_{attr}'
            df[col] = df[col].fillna(df.groupby('WEIGHTCLASS')[col].transform('mean'))
            df[col] = df[col].fillna(df[col].mean())
    
    # Fill NaN values for stance using weight-class-specific mode
    for fighter_num in [1, 2]:
        col = f'fighter{fighter_num}_stance'
        weight_class_modes = df.groupby('WEIGHTCLASS')[col].agg(get_mode)
        df[col] = df[col].fillna(df['WEIGHTCLASS'].map(weight_class_modes))
        overall_mode = df[col].mode()
        if len(overall_mode) > 0:
            df[col] = df[col].fillna(overall_mode.iloc[0])
    
    return df


def preprocess_data():
    df = combine_dataframes()
    df = fill_nan_values(df)
    #print(df.columns)
    df.drop_duplicates(inplace=True)
    return df


