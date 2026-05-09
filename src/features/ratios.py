import numpy as np

# Ratio cap: prevents near-zero denominators from creating extreme outliers
_RATIO_CAP = 10.0


def _safe_ratio(a, b):
    return (a / (b + 1e-6)).clip(upper=_RATIO_CAP)


def create_ratio_features(df):
    """
    Create ratio and difference features comparing fighter1 vs fighter2 metrics.
    Ratios are capped at 10 to prevent extreme values when denominator is near zero.
    Difference features complement ratios and are more stable when both values are near zero.
    """
    # ========== HISTORICAL RATIO FEATURES (capped) ==========
    df['win_rate_ratio'] = _safe_ratio(df['fighter1_win_rate_last_5'], df['fighter2_win_rate_last_5'])
    df['finish_rate_ratio'] = _safe_ratio(df['fighter1_finish_rate_last_5'], df['fighter2_finish_rate_last_5'])
    df['ko_rate_ratio'] = _safe_ratio(df['fighter1_ko_rate_last_5'], df['fighter2_ko_rate_last_5'])
    df['sub_rate_ratio'] = _safe_ratio(df['fighter1_sub_rate_last_5'], df['fighter2_sub_rate_last_5'])
    df['decision_rate_ratio'] = _safe_ratio(df['fighter1_decision_rate_last_5'], df['fighter2_decision_rate_last_5'])
    df['early_finish_rate_ratio'] = _safe_ratio(df['fighter1_early_finish_rate_last_5'], df['fighter2_early_finish_rate_last_5'])

    df['avg_sig_strikes_ratio'] = _safe_ratio(df['fighter1_avg_sig_strikes_last_3'], df['fighter2_avg_sig_strikes_last_3'])
    df['avg_control_time_ratio'] = _safe_ratio(df['fighter1_avg_control_time_last_3'], df['fighter2_avg_control_time_last_3'])
    df['total_fights_ratio'] = _safe_ratio(df['fighter1_total_fights'], df['fighter2_total_fights'])
    df['days_in_ufc_ratio'] = _safe_ratio(df['fighter1_days_in_ufc'], df['fighter2_days_in_ufc'])
    df['avg_finish_round_ratio'] = _safe_ratio(df['fighter1_avg_finish_round_last_5'], df['fighter2_avg_finish_round_last_5'])
    df['avg_finish_time_ratio'] = _safe_ratio(df['fighter1_avg_finish_time_last_5'], df['fighter2_avg_finish_time_last_5'])

    df['avg_takedowns_ratio'] = _safe_ratio(df['fighter1_avg_takedowns_landed_last_3'], df['fighter2_avg_takedowns_landed_last_3'])
    df['avg_KD_ratio'] = _safe_ratio(df['fighter1_avg_KD_last_3'], df['fighter2_avg_KD_last_3'])
    df['avg_head_strikes_ratio'] = _safe_ratio(df['fighter1_avg_head_landed_last_3'], df['fighter2_avg_head_landed_last_3'])
    df['avg_body_strikes_ratio'] = _safe_ratio(df['fighter1_avg_body_landed_last_3'], df['fighter2_avg_body_landed_last_3'])
    df['avg_leg_strikes_ratio'] = _safe_ratio(df['fighter1_avg_leg_landed_last_3'], df['fighter2_avg_leg_landed_last_3'])
    df['avg_distance_strikes_ratio'] = _safe_ratio(df['fighter1_avg_distance_landed_last_3'], df['fighter2_avg_distance_landed_last_3'])
    df['avg_clinch_strikes_ratio'] = _safe_ratio(df['fighter1_avg_clinch_landed_last_3'], df['fighter2_avg_clinch_landed_last_3'])
    df['avg_ground_strikes_ratio'] = _safe_ratio(df['fighter1_avg_ground_landed_last_3'], df['fighter2_avg_ground_landed_last_3'])
    df['avg_sub_att_ratio'] = _safe_ratio(df['fighter1_avg_SUB.ATT_last_3'], df['fighter2_avg_SUB.ATT_last_3'])
    df['avg_rev_ratio'] = _safe_ratio(df['fighter1_avg_REV._last_3'], df['fighter2_avg_REV._last_3'])
    df['avg_total_strikes_ratio'] = _safe_ratio(df['fighter1_avg_total_strikes_landed_last_3'], df['fighter2_avg_total_strikes_landed_last_3'])

    # ========== DIFFERENCE FEATURES ==========
    # Win rate difference (more stable than ratio when both values are near zero)
    df['win_rate_diff'] = df['fighter1_win_rate_last_5'] - df['fighter2_win_rate_last_5']
    df['finish_rate_diff'] = df['fighter1_finish_rate_last_5'] - df['fighter2_finish_rate_last_5']
    df['ko_rate_diff'] = df['fighter1_ko_rate_last_5'] - df['fighter2_ko_rate_last_5']
    df['sub_rate_diff'] = df['fighter1_sub_rate_last_5'] - df['fighter2_sub_rate_last_5']

    # Striking accuracy differences (accuracy vs volume)
    df['sig_strikes_pct_diff'] = df['fighter1_sig_strikes_pct'] - df['fighter2_sig_strikes_pct']
    df['takedown_pct_diff'] = df['fighter1_takedown_pct'] - df['fighter2_takedown_pct']

    # Defragment DataFrame after ratio features
    df = df.copy()

    return df
