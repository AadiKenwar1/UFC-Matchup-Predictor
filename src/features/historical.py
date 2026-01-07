import pandas as pd
import numpy as np
from preprocessor import parse_time_seconds
from listOfFeatures import RATE_COLS, AVG_COLS
from .helpers import categorize_method, calc_historical_avg

def create_historical_features(df):
    """
    Create historical features: win rates, averages, finish rates, etc.
    """
    # Parse OUTCOME to determine winner (L/W means fighter1 lost, W/L means fighter1 won, D/D means draw)
    df['fighter1_won'] = df['OUTCOME'].apply(lambda x: 1 if x == 'W/L' else (0 if x == 'L/W' else np.nan))
    
    # Calculate historical win rate (last 5 fights) for each fighter
    # For fighter1: win rate based on when they were fighter1
    df['fighter1_won_shifted'] = df.groupby('fighter1_name')['fighter1_won'].shift(1)
    df['fighter1_win_rate_last_5'] = df.groupby('fighter1_name')['fighter1_won_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # For fighter2: win rate based on when they were fighter2 (flip outcome: if fighter1 lost, fighter2 won)
    df['fighter2_won'] = df['OUTCOME'].apply(lambda x: 1 if x == 'L/W' else (0 if x == 'W/L' else np.nan))
    df['fighter2_won_shifted'] = df.groupby('fighter2_name')['fighter2_won'].shift(1)
    df['fighter2_win_rate_last_5'] = df.groupby('fighter2_name')['fighter2_won_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Fill NaN win rates (first fights) with 0.5 (neutral)
    df['fighter1_win_rate_last_5'] = df['fighter1_win_rate_last_5'].fillna(0.5)
    df['fighter2_win_rate_last_5'] = df['fighter2_win_rate_last_5'].fillna(0.5)
    
    # Calculate average sig strikes landed in last 3 fights
    fighter1_sig_avg = df.groupby('fighter1_name')['fighter1_sig_strikes_landed'].apply(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    df['fighter1_avg_sig_strikes_last_3'] = fighter1_sig_avg.values
    
    fighter2_sig_avg = df.groupby('fighter2_name')['fighter2_sig_strikes_landed'].apply(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    df['fighter2_avg_sig_strikes_last_3'] = fighter2_sig_avg.values
    
    # Fill NaN with column mean
    df['fighter1_avg_sig_strikes_last_3'] = df['fighter1_avg_sig_strikes_last_3'].fillna(df['fighter1_avg_sig_strikes_last_3'].mean())
    df['fighter2_avg_sig_strikes_last_3'] = df['fighter2_avg_sig_strikes_last_3'].fillna(df['fighter2_avg_sig_strikes_last_3'].mean())
    
    # Calculate average control time in last 3 fights
    fighter1_ctrl_avg = df.groupby('fighter1_name')['fighter1_control_time_sec'].apply(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    df['fighter1_avg_control_time_last_3'] = fighter1_ctrl_avg.values
    
    fighter2_ctrl_avg = df.groupby('fighter2_name')['fighter2_control_time_sec'].apply(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    df['fighter2_avg_control_time_last_3'] = fighter2_ctrl_avg.values
    
    # Fill NaN with column mean
    df['fighter1_avg_control_time_last_3'] = df['fighter1_avg_control_time_last_3'].fillna(df['fighter1_avg_control_time_last_3'].mean())
    df['fighter2_avg_control_time_last_3'] = df['fighter2_avg_control_time_last_3'].fillna(df['fighter2_avg_control_time_last_3'].mean())
    
    # Calculate historical averages for all fight statistics
    historical_stats = [
        'total_strikes_landed', 'ground_landed', 'KD', 'head_landed', 
        'body_landed', 'leg_landed', 'distance_landed', 'clinch_landed',
        'takedowns_landed', 'SUB.ATT', 'REV.'
    ]
    
    # Calculate historical averages for all fight statistics
    for stat in historical_stats:
        df = calc_historical_avg(df, 1, stat)
        df = calc_historical_avg(df, 2, stat)
    
    # Defragment DataFrame after historical averages
    df = df.copy()
    
    # Parse ROUND to numeric
    df['ROUND_numeric'] = pd.to_numeric(df['ROUND'], errors='coerce')
    
    # Parse TIME to seconds
    df['TIME_seconds'] = df['TIME'].apply(parse_time_seconds)
    
    # Create columns showing how each fighter won (METHOD if they won, None otherwise)
    df['fighter1_win_method'] = df.apply(lambda row: categorize_method(row['METHOD']) if row['OUTCOME'] == 'W/L' else None, axis=1)
    df['fighter2_win_method'] = df.apply(lambda row: categorize_method(row['METHOD']) if row['OUTCOME'] == 'L/W' else None, axis=1)
    
    # Create columns showing finish round/time for wins
    df['fighter1_win_round'] = df.apply(lambda row: row['ROUND_numeric'] if row['OUTCOME'] == 'W/L' else None, axis=1)
    df['fighter2_win_round'] = df.apply(lambda row: row['ROUND_numeric'] if row['OUTCOME'] == 'L/W' else None, axis=1)
    
    df['fighter1_win_time_sec'] = df.apply(lambda row: row['TIME_seconds'] if row['OUTCOME'] == 'W/L' else None, axis=1)
    df['fighter2_win_time_sec'] = df.apply(lambda row: row['TIME_seconds'] if row['OUTCOME'] == 'L/W' else None, axis=1)
    
    # Create binary indicators for finish types (for rolling calculations)
    df['fighter1_win_finish'] = df['fighter1_win_method'].apply(lambda x: 1 if x in ['KO/TKO', 'Submission'] else (0 if pd.notna(x) else np.nan))
    df['fighter2_win_finish'] = df['fighter2_win_method'].apply(lambda x: 1 if x in ['KO/TKO', 'Submission'] else (0 if pd.notna(x) else np.nan))
    
    df['fighter1_win_ko'] = df['fighter1_win_method'].apply(lambda x: 1 if x == 'KO/TKO' else (0 if pd.notna(x) else np.nan))
    df['fighter2_win_ko'] = df['fighter2_win_method'].apply(lambda x: 1 if x == 'KO/TKO' else (0 if pd.notna(x) else np.nan))
    
    df['fighter1_win_sub'] = df['fighter1_win_method'].apply(lambda x: 1 if x == 'Submission' else (0 if pd.notna(x) else np.nan))
    df['fighter2_win_sub'] = df['fighter2_win_method'].apply(lambda x: 1 if x == 'Submission' else (0 if pd.notna(x) else np.nan))
    
    df['fighter1_win_decision'] = df['fighter1_win_method'].apply(lambda x: 1 if x == 'Decision' else (0 if pd.notna(x) else np.nan))
    df['fighter2_win_decision'] = df['fighter2_win_method'].apply(lambda x: 1 if x == 'Decision' else (0 if pd.notna(x) else np.nan))
    
    df['fighter1_win_early'] = df['fighter1_win_round'].apply(lambda x: 1 if pd.notna(x) and x <= 2 else (0 if pd.notna(x) else np.nan))
    df['fighter2_win_early'] = df['fighter2_win_round'].apply(lambda x: 1 if pd.notna(x) and x <= 2 else (0 if pd.notna(x) else np.nan))
    
    # Calculate historical finish rates (last 5 wins) for each fighter using shift and rolling
    df['fighter1_win_finish_shifted'] = df.groupby('fighter1_name')['fighter1_win_finish'].shift(1)
    df['fighter1_finish_rate_last_5'] = df.groupby('fighter1_name')['fighter1_win_finish_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    df['fighter2_win_finish_shifted'] = df.groupby('fighter2_name')['fighter2_win_finish'].shift(1)
    df['fighter2_finish_rate_last_5'] = df.groupby('fighter2_name')['fighter2_win_finish_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Calculate KO/TKO rate (last 5 wins)
    df['fighter1_win_ko_shifted'] = df.groupby('fighter1_name')['fighter1_win_ko'].shift(1)
    df['fighter1_ko_rate_last_5'] = df.groupby('fighter1_name')['fighter1_win_ko_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    df['fighter2_win_ko_shifted'] = df.groupby('fighter2_name')['fighter2_win_ko'].shift(1)
    df['fighter2_ko_rate_last_5'] = df.groupby('fighter2_name')['fighter2_win_ko_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Calculate Submission rate (last 5 wins)
    df['fighter1_win_sub_shifted'] = df.groupby('fighter1_name')['fighter1_win_sub'].shift(1)
    df['fighter1_sub_rate_last_5'] = df.groupby('fighter1_name')['fighter1_win_sub_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    df['fighter2_win_sub_shifted'] = df.groupby('fighter2_name')['fighter2_win_sub'].shift(1)
    df['fighter2_sub_rate_last_5'] = df.groupby('fighter2_name')['fighter2_win_sub_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Calculate Decision rate (last 5 wins)
    df['fighter1_win_decision_shifted'] = df.groupby('fighter1_name')['fighter1_win_decision'].shift(1)
    df['fighter1_decision_rate_last_5'] = df.groupby('fighter1_name')['fighter1_win_decision_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    df['fighter2_win_decision_shifted'] = df.groupby('fighter2_name')['fighter2_win_decision'].shift(1)
    df['fighter2_decision_rate_last_5'] = df.groupby('fighter2_name')['fighter2_win_decision_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Calculate average finish round (last 5 wins)
    df['fighter1_win_round_shifted'] = df.groupby('fighter1_name')['fighter1_win_round'].shift(1)
    df['fighter1_avg_finish_round_last_5'] = df.groupby('fighter1_name')['fighter1_win_round_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    df['fighter2_win_round_shifted'] = df.groupby('fighter2_name')['fighter2_win_round'].shift(1)
    df['fighter2_avg_finish_round_last_5'] = df.groupby('fighter2_name')['fighter2_win_round_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Calculate early finish rate (rounds 1-2) (last 5 wins)
    df['fighter1_win_early_shifted'] = df.groupby('fighter1_name')['fighter1_win_early'].shift(1)
    df['fighter1_early_finish_rate_last_5'] = df.groupby('fighter1_name')['fighter1_win_early_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    df['fighter2_win_early_shifted'] = df.groupby('fighter2_name')['fighter2_win_early'].shift(1)
    df['fighter2_early_finish_rate_last_5'] = df.groupby('fighter2_name')['fighter2_win_early_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Calculate average finish time in seconds (last 5 wins)
    df['fighter1_win_time_sec_shifted'] = df.groupby('fighter1_name')['fighter1_win_time_sec'].shift(1)
    df['fighter1_avg_finish_time_last_5'] = df.groupby('fighter1_name')['fighter1_win_time_sec_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    df['fighter2_win_time_sec_shifted'] = df.groupby('fighter2_name')['fighter2_win_time_sec'].shift(1)
    df['fighter2_avg_finish_time_last_5'] = df.groupby('fighter2_name')['fighter2_win_time_sec_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Fill NaN values with defaults (0 for rates, median for averages)
    for col in RATE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    for col in AVG_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else (2.5 if 'round' in col else 180))
    
    # Defragment DataFrame after win/finish rate calculations
    df = df.copy()
    
    return df

