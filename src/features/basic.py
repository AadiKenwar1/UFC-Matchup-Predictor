import pandas as pd
import numpy as np

def create_basic_features(df):
    """
    Create basic features: temporal, age, differences, days since last fight, etc.
    """
    # Extract temporal features (DATE is already datetime from preprocessor)
    df['month'] = df['DATE'].dt.month
    
    # Calculate ages at fight date (DOB is already datetime from preprocessor)
    df['fighter1_age'] = (df['DATE'] - df['fighter1_dob']).dt.days / 365.25
    df['fighter2_age'] = (df['DATE'] - df['fighter2_dob']).dt.days / 365.25
    
    # Create age_unknown flags
    df['fighter1_age_unknown'] = df['fighter1_age'].isna().astype(int)
    df['fighter2_age_unknown'] = df['fighter2_age'].isna().astype(int)
    
    # Transform TIME FORMAT to is_title_fight (1 for 5 rounds, 0 for 3 rounds)
    df['is_title_fight'] = df['TIME FORMAT'].str.contains('5 Rnd').astype(int)
    
    # Create stance matchup feature
    df['stance_matchup'] = df['fighter1_stance'].fillna('Unknown') + '_vs_' + df['fighter2_stance'].fillna('Unknown')
    
    # Sort by date for temporal calculations
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # Calculate difference features
    df['height_diff'] = df['fighter1_height'] - df['fighter2_height']
    df['weight_diff'] = df['fighter1_weight'] - df['fighter2_weight']
    df['reach_diff'] = df['fighter1_reach'] - df['fighter2_reach']
    df['age_diff'] = df['fighter1_age'] - df['fighter2_age']
    df['age_diff_unknown'] = (df['fighter1_age_unknown'] | df['fighter2_age_unknown']).astype(int)
    
    # Calculate days since last fight for each fighter
    df['fighter1_days_since_last_fight'] = df.groupby('fighter1_name')['DATE'].diff().dt.days
    df['fighter2_days_since_last_fight'] = df.groupby('fighter2_name')['DATE'].diff().dt.days
    
    # Fill NaN for first fights with median
    median_days_f1 = df['fighter1_days_since_last_fight'].median()
    median_days_f2 = df['fighter2_days_since_last_fight'].median()
    df['fighter1_days_since_last_fight'] = df['fighter1_days_since_last_fight'].fillna(median_days_f1 if not pd.isna(median_days_f1) else 180)
    df['fighter2_days_since_last_fight'] = df['fighter2_days_since_last_fight'].fillna(median_days_f2 if not pd.isna(median_days_f2) else 180)
    
    # Calculate total fights up to this point for each fighter
    df['fighter1_total_fights'] = df.groupby('fighter1_name').cumcount() + 1
    df['fighter2_total_fights'] = df.groupby('fighter2_name').cumcount() + 1
    
    # Calculate days since first UFC fight (career length in days)
    df['fighter1_first_fight_date'] = df.groupby('fighter1_name')['DATE'].transform('min')
    df['fighter1_days_in_ufc'] = (df['DATE'] - df['fighter1_first_fight_date']).dt.days
    df['fighter2_first_fight_date'] = df.groupby('fighter2_name')['DATE'].transform('min')
    df['fighter2_days_in_ufc'] = (df['DATE'] - df['fighter2_first_fight_date']).dt.days
    # Drop intermediate columns
    df = df.drop(columns=['fighter1_first_fight_date', 'fighter2_first_fight_date'])
    
    return df

