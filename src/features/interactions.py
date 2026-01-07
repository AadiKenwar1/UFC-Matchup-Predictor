def create_interaction_features(df):
    """
    Create interaction features: physical attribute interactions.
    """
    # ========== INTERACTION FEATURES ==========
    # Fill missing height/weight/reach for interaction features
    df['fighter1_height_filled'] = df['fighter1_height'].fillna(df['fighter1_height'].median())
    df['fighter2_height_filled'] = df['fighter2_height'].fillna(df['fighter2_height'].median())
    df['fighter1_weight_filled'] = df['fighter1_weight'].fillna(df['fighter1_weight'].median())
    df['fighter2_weight_filled'] = df['fighter2_weight'].fillna(df['fighter2_weight'].median())
    df['fighter1_reach_filled'] = df['fighter1_reach'].fillna(df['fighter1_reach'].median())
    df['fighter2_reach_filled'] = df['fighter2_reach'].fillna(df['fighter2_reach'].median())
    
    # Reach advantage x striking ability
    df['reach_advantage_x_striking'] = df['reach_diff'] * df['avg_sig_strikes_ratio']
    
    # Age x experience difference
    df['age_x_experience_diff'] = df['age_diff'] * (df['fighter1_total_fights'] - df['fighter2_total_fights'])
    
    # Size advantage (height * weight)
    df['size_advantage_f1'] = df['fighter1_height_filled'] * df['fighter1_weight_filled']
    df['size_advantage_f2'] = df['fighter2_height_filled'] * df['fighter2_weight_filled']
    df['size_advantage_diff'] = df['size_advantage_f1'] - df['size_advantage_f2']
    
    # Power advantage (weight * reach)
    df['power_advantage_f1'] = df['fighter1_weight_filled'] * df['fighter1_reach_filled']
    df['power_advantage_f2'] = df['fighter2_weight_filled'] * df['fighter2_reach_filled']
    df['power_advantage_diff'] = df['power_advantage_f1'] - df['power_advantage_f2']
    
    # Reach x win rate interaction
    df['reach_x_win_rate'] = df['reach_diff'] * df['win_rate_ratio']
    
    # Age x momentum interaction
    df['age_x_momentum'] = df['age_diff'] * df['momentum_diff']
    
    # Size x finish rate interaction
    df['size_x_finish_rate'] = df['size_advantage_diff'] * df['finish_rate_ratio']
    
    # Defragment DataFrame after interaction features
    df = df.copy()
    
    return df

