import pandas as pd

def calc_streak(series, win=True):
    """Calculate win or loss streak from a series of results."""
    # Fill NaNs with a neutral value (e.g., 0.5) for streak calculation
    s = series.fillna(0.5) 
    if win:
        return s.groupby((s != 1).cumsum()).cumcount() * (s == 1)
    else:
        return s.groupby((s != 0).cumsum()).cumcount() * (s == 0)

def create_momentum_features(df):
    """
    Create momentum features: career win rate, momentum, and win/loss streaks.
    """
    # ========== MOMENTUM FEATURES ==========
    # Career win rate (overall, not just last 5)
    df['fighter1_career_win_rate'] = df.groupby('fighter1_name')['fighter1_won_shifted'].transform(
        lambda x: x.expanding(min_periods=1).mean()
    ).fillna(0.5)
    df['fighter2_career_win_rate'] = df.groupby('fighter2_name')['fighter2_won_shifted'].transform(
        lambda x: x.expanding(min_periods=1).mean()
    ).fillna(0.5)
    
    # Momentum: recent form vs career average (positive = improving, negative = declining)
    df['fighter1_momentum'] = df['fighter1_win_rate_last_5'] - df['fighter1_career_win_rate']
    df['fighter2_momentum'] = df['fighter2_win_rate_last_5'] - df['fighter2_career_win_rate']
    df['momentum_diff'] = df['fighter1_momentum'] - df['fighter2_momentum']
    
    # Fill NaN momentum (first fight) with 0
    df['fighter1_momentum'] = df['fighter1_momentum'].fillna(0)
    df['fighter2_momentum'] = df['fighter2_momentum'].fillna(0)
    df['momentum_diff'] = df['momentum_diff'].fillna(0)
    
    # Defragment DataFrame after momentum features
    df = df.copy()
    
    # ========== WIN/LOSS STREAK FEATURES ==========
    # Use shifted results to avoid data leakage
    df['fighter1_result'] = df['fighter1_won_shifted']
    df['fighter2_result'] = df['fighter2_won_shifted']
    
    df['fighter1_win_streak'] = df.groupby('fighter1_name')['fighter1_result'].transform(
        lambda x: calc_streak(x, win=True)
    ).fillna(0)
    df['fighter2_win_streak'] = df.groupby('fighter2_name')['fighter2_result'].transform(
        lambda x: calc_streak(x, win=True)
    ).fillna(0)
    
    df['fighter1_loss_streak'] = df.groupby('fighter1_name')['fighter1_result'].transform(
        lambda x: calc_streak(x, win=False)
    ).fillna(0)
    df['fighter2_loss_streak'] = df.groupby('fighter2_name')['fighter2_result'].transform(
        lambda x: calc_streak(x, win=False)
    ).fillna(0)
    
    # Streak difference
    df['win_streak_diff'] = df['fighter1_win_streak'] - df['fighter2_win_streak']
    df['loss_streak_diff'] = df['fighter1_loss_streak'] - df['fighter2_loss_streak']
    
    # Drop intermediate columns
    df = df.drop(columns=['fighter1_result', 'fighter2_result'], errors='ignore')
    
    # Defragment DataFrame after streak features
    df = df.copy()
    
    return df

