import pandas as pd

def create_consistency_features(df):
    """
    Create consistency features: variance/standard deviation metrics.
    """
    # ========== CONSISTENCY FEATURES ==========
    # Performance consistency metrics (variance/standard deviation)
    # Lower variance = more consistent performance
    
    # Win rate consistency (variance in win/loss results over last 5 fights)
    df['fighter1_win_rate_std'] = df.groupby('fighter1_name')['fighter1_won_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=2).std()
    ).fillna(0.5)  # Default to 0.5 (moderate variance) for fighters with < 2 fights
    df['fighter2_win_rate_std'] = df.groupby('fighter2_name')['fighter2_won_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=2).std()
    ).fillna(0.5)
    df['win_rate_consistency_diff'] = df['fighter1_win_rate_std'] - df['fighter2_win_rate_std']
    
    # Strike output consistency (variance in strikes landed over last 5 fights)
    df['fighter1_strike_output_std'] = df.groupby('fighter1_name')['fighter1_sig_strikes_landed'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=2).std()
    ).fillna(df['fighter1_sig_strikes_landed'].std() if df['fighter1_sig_strikes_landed'].std() > 0 else 50)
    df['fighter2_strike_output_std'] = df.groupby('fighter2_name')['fighter2_sig_strikes_landed'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=2).std()
    ).fillna(df['fighter2_sig_strikes_landed'].std() if df['fighter2_sig_strikes_landed'].std() > 0 else 50)
    df['strike_output_consistency_diff'] = df['fighter1_strike_output_std'] - df['fighter2_strike_output_std']
    
    # Finish rate consistency (variance in finishing ability - binary: finish or not)
    # Use finish rate variance over last 5 wins
    df['fighter1_finish_consistency'] = df.groupby('fighter1_name')['fighter1_win_finish_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=2).std()
    ).fillna(0.5)
    df['fighter2_finish_consistency'] = df.groupby('fighter2_name')['fighter2_win_finish_shifted'].transform(
        lambda x: x.rolling(window=5, min_periods=2).std()
    ).fillna(0.5)
    df['finish_consistency_diff'] = df['fighter1_finish_consistency'] - df['fighter2_finish_consistency']
    
    # Control time consistency (variance in control time over last 5 fights)
    df['fighter1_control_time_std'] = df.groupby('fighter1_name')['fighter1_control_time_sec'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=2).std()
    ).fillna(df['fighter1_control_time_sec'].std() if df['fighter1_control_time_sec'].std() > 0 else 30)
    df['fighter2_control_time_std'] = df.groupby('fighter2_name')['fighter2_control_time_sec'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=2).std()
    ).fillna(df['fighter2_control_time_sec'].std() if df['fighter2_control_time_sec'].std() > 0 else 30)
    df['control_time_consistency_diff'] = df['fighter1_control_time_std'] - df['fighter2_control_time_std']
    
    # Takedown activity consistency (variance in takedown attempts over last 5 fights)
    df['fighter1_takedown_std'] = df.groupby('fighter1_name')['fighter1_takedowns_attempted'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=2).std()
    ).fillna(df['fighter1_takedowns_attempted'].std() if df['fighter1_takedowns_attempted'].std() > 0 else 2)
    df['fighter2_takedown_std'] = df.groupby('fighter2_name')['fighter2_takedowns_attempted'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=2).std()
    ).fillna(df['fighter2_takedowns_attempted'].std() if df['fighter2_takedowns_attempted'].std() > 0 else 2)
    df['takedown_consistency_diff'] = df['fighter1_takedown_std'] - df['fighter2_takedown_std']
    
    # Defragment DataFrame after consistency features
    df = df.copy()
    
    return df

