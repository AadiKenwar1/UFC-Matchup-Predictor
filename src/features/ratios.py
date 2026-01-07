def create_ratio_features(df):
    """
    Create ratio features comparing fighter1 vs fighter2 metrics.
    """
    # ========== HISTORICAL RATIO FEATURES ==========
    # Calculate ratios between fighters for various metrics
    df['win_rate_ratio'] = df['fighter1_win_rate_last_5'] / (df['fighter2_win_rate_last_5'] + 1e-6)
    df['finish_rate_ratio'] = df['fighter1_finish_rate_last_5'] / (df['fighter2_finish_rate_last_5'] + 1e-6)
    df['ko_rate_ratio'] = df['fighter1_ko_rate_last_5'] / (df['fighter2_ko_rate_last_5'] + 1e-6)
    df['sub_rate_ratio'] = df['fighter1_sub_rate_last_5'] / (df['fighter2_sub_rate_last_5'] + 1e-6)
    df['decision_rate_ratio'] = df['fighter1_decision_rate_last_5'] / (df['fighter2_decision_rate_last_5'] + 1e-6)
    df['early_finish_rate_ratio'] = df['fighter1_early_finish_rate_last_5'] / (df['fighter2_early_finish_rate_last_5'] + 1e-6)
    
    df['avg_sig_strikes_ratio'] = df['fighter1_avg_sig_strikes_last_3'] / (df['fighter2_avg_sig_strikes_last_3'] + 1e-6)
    df['avg_control_time_ratio'] = df['fighter1_avg_control_time_last_3'] / (df['fighter2_avg_control_time_last_3'] + 1e-6)
    df['total_fights_ratio'] = df['fighter1_total_fights'] / (df['fighter2_total_fights'] + 1e-6)
    df['days_in_ufc_ratio'] = df['fighter1_days_in_ufc'] / (df['fighter2_days_in_ufc'] + 1e-6)
    df['avg_finish_round_ratio'] = df['fighter1_avg_finish_round_last_5'] / (df['fighter2_avg_finish_round_last_5'] + 1e-6)
    df['avg_finish_time_ratio'] = df['fighter1_avg_finish_time_last_5'] / (df['fighter2_avg_finish_time_last_5'] + 1e-6)
    
    df['avg_takedowns_ratio'] = df['fighter1_avg_takedowns_landed_last_3'] / (df['fighter2_avg_takedowns_landed_last_3'] + 1e-6)
    df['avg_KD_ratio'] = df['fighter1_avg_KD_last_3'] / (df['fighter2_avg_KD_last_3'] + 1e-6)
    df['avg_head_strikes_ratio'] = df['fighter1_avg_head_landed_last_3'] / (df['fighter2_avg_head_landed_last_3'] + 1e-6)
    df['avg_body_strikes_ratio'] = df['fighter1_avg_body_landed_last_3'] / (df['fighter2_avg_body_landed_last_3'] + 1e-6)
    df['avg_leg_strikes_ratio'] = df['fighter1_avg_leg_landed_last_3'] / (df['fighter2_avg_leg_landed_last_3'] + 1e-6)
    df['avg_distance_strikes_ratio'] = df['fighter1_avg_distance_landed_last_3'] / (df['fighter2_avg_distance_landed_last_3'] + 1e-6)
    df['avg_clinch_strikes_ratio'] = df['fighter1_avg_clinch_landed_last_3'] / (df['fighter2_avg_clinch_landed_last_3'] + 1e-6)
    df['avg_ground_strikes_ratio'] = df['fighter1_avg_ground_landed_last_3'] / (df['fighter2_avg_ground_landed_last_3'] + 1e-6)
    df['avg_sub_att_ratio'] = df['fighter1_avg_SUB.ATT_last_3'] / (df['fighter2_avg_SUB.ATT_last_3'] + 1e-6)
    df['avg_rev_ratio'] = df['fighter1_avg_REV._last_3'] / (df['fighter2_avg_REV._last_3'] + 1e-6)
    df['avg_total_strikes_ratio'] = df['fighter1_avg_total_strikes_landed_last_3'] / (df['fighter2_avg_total_strikes_landed_last_3'] + 1e-6)
    
    # Defragment DataFrame after ratio features
    df = df.copy()
    
    return df

