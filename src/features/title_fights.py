import pandas as pd
import numpy as np

def create_title_fight_features(df):
    """
    Create title fight-related features:
    - Number of title fights (cumulative)
    - Days since last title fight
    - Is current champion (last fight was title fight and they won)
    """
    # ========== NUMBER OF TITLE FIGHTS ==========
    # Calculate cumulative number of title fights (excluding current fight)
    df['fighter1_is_title_shifted'] = df.groupby('fighter1_name')['is_title_fight'].shift(1)
    df['fighter1_num_title_fights'] = df.groupby('fighter1_name')['fighter1_is_title_shifted'].transform(
        lambda x: x.fillna(0).cumsum()
    )
    
    df['fighter2_is_title_shifted'] = df.groupby('fighter2_name')['is_title_fight'].shift(1)
    df['fighter2_num_title_fights'] = df.groupby('fighter2_name')['fighter2_is_title_shifted'].transform(
        lambda x: x.fillna(0).cumsum()
    )
    
    # Difference and ratio features for title fights
    df['title_fights_diff'] = df['fighter1_num_title_fights'] - df['fighter2_num_title_fights']
    df['title_fights_ratio'] = df['fighter1_num_title_fights'] / (df['fighter2_num_title_fights'] + 1)
    
    # ========== DAYS SINCE LAST TITLE FIGHT ==========
    # For fighter1: find the date of their last title fight (before current fight)
    # Shift to exclude current fight, then use ffill to carry forward last title fight date
    df['fighter1_title_fight_date'] = df.groupby('fighter1_name', group_keys=False).apply(
        lambda x: x['DATE'].where(x['is_title_fight'].shift(1) == 1)
    )
    df['fighter1_last_title_fight_date'] = df.groupby('fighter1_name')['fighter1_title_fight_date'].transform(
        lambda x: x.ffill()
    )
    
    df['fighter2_title_fight_date'] = df.groupby('fighter2_name', group_keys=False).apply(
        lambda x: x['DATE'].where(x['is_title_fight'].shift(1) == 1)
    )
    df['fighter2_last_title_fight_date'] = df.groupby('fighter2_name')['fighter2_title_fight_date'].transform(
        lambda x: x.ffill()
    )
    
    # Calculate days since last title fight
    df['fighter1_days_since_last_title_fight'] = (df['DATE'] - df['fighter1_last_title_fight_date']).dt.days
    df['fighter2_days_since_last_title_fight'] = (df['DATE'] - df['fighter2_last_title_fight_date']).dt.days
    
    # Fill NaN with median (for fighters who never fought for a title)
    median_days_f1 = df['fighter1_days_since_last_title_fight'].median()
    median_days_f2 = df['fighter2_days_since_last_title_fight'].median()
    df['fighter1_days_since_last_title_fight'] = df['fighter1_days_since_last_title_fight'].fillna(
        median_days_f1 if not pd.isna(median_days_f1) else 365
    )
    df['fighter2_days_since_last_title_fight'] = df['fighter2_days_since_last_title_fight'].fillna(
        median_days_f2 if not pd.isna(median_days_f2) else 365
    )
    
    # Difference feature
    df['days_since_last_title_fight_diff'] = (
        df['fighter1_days_since_last_title_fight'] - df['fighter2_days_since_last_title_fight']
    )
    
    # ========== IS CURRENT CHAMPION ==========
    # Current champion: last fight was a title fight AND they won it
    # fighter1_won_shifted and fighter2_won_shifted are already created in historical.py
    
    df['fighter1_last_fight_was_title'] = df.groupby('fighter1_name')['is_title_fight'].shift(1).fillna(0)
    df['fighter1_is_current_champion'] = (
        (df['fighter1_last_fight_was_title'] == 1) & 
        (df['fighter1_won_shifted'] == 1)
    ).astype(int)
    
    df['fighter2_last_fight_was_title'] = df.groupby('fighter2_name')['is_title_fight'].shift(1).fillna(0)
    df['fighter2_is_current_champion'] = (
        (df['fighter2_last_fight_was_title'] == 1) & 
        (df['fighter2_won_shifted'] == 1)
    ).astype(int)
    
    # Difference feature (1 if fighter1 is champion and fighter2 is not, -1 if opposite, 0 if both/neither)
    df['champion_diff'] = df['fighter1_is_current_champion'] - df['fighter2_is_current_champion']
    
    # Both champions flag (indicates championship bout)
    df['both_champions'] = (
        (df['fighter1_is_current_champion'] == 1) & 
        (df['fighter2_is_current_champion'] == 1)
    ).astype(int)
    
    # Drop intermediate columns
    df = df.drop(columns=[
        'fighter1_is_title_shifted', 'fighter2_is_title_shifted',
        'fighter1_title_fight_date', 'fighter2_title_fight_date',
        'fighter1_last_title_fight_date', 'fighter2_last_title_fight_date',
        'fighter1_last_fight_was_title', 'fighter2_last_fight_was_title'
    ])
    
    # Defragment DataFrame after title fight features
    df = df.copy()
    
    return df

