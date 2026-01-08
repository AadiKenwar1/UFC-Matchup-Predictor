# UFC Fight Predictor

An end-to-end machine learning system that predicts UFC fight outcomes using historical fighter data, performance metrics, and matchup dynamics. The system achieves 63% accuracy (ROC-AUC: 0.65) in predicting fight winners—significantly outperforming random chance in a highly unpredictable sport.

**Live Demo:** [Deployed on Render](https://ufc-matchup-predictor.onrender.com/)

## Features

- Real-time fight outcome predictions with win probability percentages
- Interactive web interface for fighter selection and visualization
- RESTful API for programmatic access
- Historical fighter performance analysis

## Tech Stack

- **Backend:** Python, FastAPI, XGBoost
- **Frontend:** HTML, CSS, JavaScript
- **ML Libraries:** scikit-learn, pandas, numpy
- **Deployment:** Render

## ML Pipeline

### 1. Exploratory Data Analysis (EDA)

Initial analysis of 8,400+ historical fights across multiple data sources:
- Fight outcomes, methods, and round distributions
- Weight class and location patterns
- Fighter attribute distributions (height, weight, reach, stance)
- Missing value analysis
- Temporal trends in fight frequency

This analysis informed feature engineering decisions and highlighted data quality considerations.

### 2. Data Preprocessing

**Data Integration:**
- Combined 6 CSV files: event details, fight results, fight stats, and fighter attributes
- Merged data on EVENT, BOUT, and FIGHTER identifiers
- Extracted fighter names from bout strings ("Fighter1 vs. Fighter2")

**Data Parsing:**
- Height: `5' 9"` → 69 inches
- Weight: `125 lbs.` → 125
- Reach: `68"` → 68
- Time: `4:32` → 272 seconds
- Percentages: `65%` → 0.65
- Fractions: `17 of 26` → (17, 26) landed/attempted

**Missing Value Handling:**
- Weight-class-specific imputation for physical attributes (height, weight, reach)
- Weight-class-specific mode for categorical variables (stance)
- Fallback to overall mean/mode when weight-class data unavailable

### 3. Feature Engineering

Features were engineered in a specific order, with each category building on previous transformations:

#### Basic Features (`basic.py`)
- **Temporal:** Month of fight (seasonal patterns)
- **Age:** Calculated from date of birth at fight time
- **Physical Differences:** Height, weight, reach, and age differentials between fighters
- **Career Metrics:** Days since last fight, total fights, days in UFC (experience indicators)
- **Matchup Context:** Stance matchup, title fight indicator (5-round vs 3-round)

#### Historical Features (`historical.py`)
- **Win Rates:** Last 5 fights (recent form indicator)
- **Performance Averages:** Rolling averages over last 3 fights for:
  - Significant strikes, total strikes, takedowns
  - Control time, knockdowns, submission attempts
  - Strike distribution (head, body, leg, distance, clinch, ground)
- **Finish Patterns:** KO/TKO rate, submission rate, decision rate, early finish rate (rounds 1-2)
- **Finish Timing:** Average finish round and time for wins

**Why:** Recent performance is more predictive than career-long averages. Rolling windows capture form while smoothing outliers.

#### Ratio Features (`ratios.py`)
- **Comparative Metrics:** Fighter1 vs Fighter2 ratios for:
  - Win rates, finish rates, striking averages
  - Experience (total fights, days in UFC)
  - Performance consistency metrics

**Why:** Relative advantages matter more than absolute values. A fighter with 60% win rate facing a 40% opponent is different than both at 50%.

#### Momentum Features (`momentum.py`)
- **Career Win Rate:** Overall win percentage (expanding window)
- **Momentum:** Recent form (last 5) minus career average
  - Positive = improving, Negative = declining
- **Streaks:** Current win/loss streaks

**Why:** Captures whether fighters are trending up or down, which affects performance beyond raw statistics.

#### Interaction Features (`interactions.py`)
- **Physical × Performance:** Reach advantage × striking ability
- **Size Metrics:** Height × weight (size advantage), weight × reach (power advantage)
- **Experience Interactions:** Age × experience differential
- **Contextual:** Reach × win rate, size × finish rate

**Why:** Physical attributes alone don't determine outcomes—how they interact with skills matters. A reach advantage is more valuable for strong strikers.

#### Consistency Features (`consistency.py`)
- **Performance Variance:** Standard deviation of:
  - Win/loss results (consistency)
  - Strike output (reliability)
  - Finish ability (predictability)
  - Control time and takedown activity

**Why:** Consistent fighters are more predictable. High variance indicates unpredictable performance.

#### Encoding Features (`encoding.py`)
- **Target Variable:** Binary classification (fighter1 wins = 1, fighter2 wins = 0)
- **Categorical Encoding:** One-hot encoding for:
  - Referee (different refereeing styles)
  - Weight class (different competitive environments)
  - Stance matchup (Orthodox vs Southpaw dynamics)

**Why:** Categorical variables need encoding for tree-based models. One-hot with drop_first prevents multicollinearity.

### 4. Model Selection

**XGBoost** was chosen for several reasons:
- **Tabular Data Excellence:** Tree-based models outperform neural networks on structured, tabular data
- **Feature Interactions:** Automatically captures non-linear relationships and feature interactions
- **Interpretability:** Feature importance scores provide insights
- **Performance:** Fast training and prediction, handles missing values well
- **Regularization:** Built-in L1/L2 regularization prevents overfitting

**Hyperparameters:**
- `n_estimators: 140`, `max_depth: 4`, `learning_rate: 0.01`
- `subsample: 0.9`, `colsample_bytree: 0.9` (prevents overfitting)
- `reg_lambda: 1.15`, `reg_alpha: 0.05` (L2/L1 regularization)
- `scale_pos_weight: 0.6` (handles class imbalance)

### 5. Data Splitting Strategy

**Temporal Splitting** (not random):
- **Train:** Fights before 2020-01-01 (5,379 samples)
- **Validation:** Fights from 2020-01-01 to 2024-01-01 (1,966 samples)
- **Test:** Fights from 2024-01-01 onwards (1,042 samples)

**Why Temporal:**
- **Prevents Data Leakage:** Random splits can use future data to predict past fights
- **Realistic Evaluation:** Simulates real-world scenario where we predict future fights using only historical data
- **Temporal Dependencies:** Fighters evolve over time; model must generalize to future performance

**Leakage Prevention:**
- All historical features use `shift(1)` to ensure only past data informs current predictions
- Rolling averages calculated from previous fights only
- Win rates exclude the current fight being predicted

### 6. Training Process

**Model Calibration:**
- `scale_pos_weight: 0.6` (< 1.0) penalizes the positive class (fighter1 wins) more heavily
- Makes the model more conservative about predicting wins, **reducing false positives** (predicting wins when fighter1 actually loses)
- Improves prediction reliability by requiring stronger evidence before predicting a win

**Training Approach:**
- **Development:** `train.py` uses temporal splits for validation and hyperparameter tuning
- **Production:** `trainFinal.py` trains on all available data (no validation split) for maximum model performance

**Evaluation Metrics:**
- **Accuracy:** 62% (vs 50% random baseline)
- **ROC-AUC:** 0.65 (measures ability to distinguish winners from losers)
- **Context:** Strong performance given UFC's unpredictability—even favorites lose ~40% of the time

## Project Structure

```
UFC Predictor/
├── data/                 # CSV data files (events, results, stats, fighters)
├── src/
│   ├── backend/         # FastAPI server (api.py, run_api.py)
│   ├── features/        # Feature engineering modules
│   │   ├── basic.py     # Basic features (age, differences, etc.)
│   │   ├── historical.py # Historical performance metrics
│   │   ├── ratios.py    # Fighter comparison ratios
│   │   ├── momentum.py  # Career momentum and streaks
│   │   ├── interactions.py # Feature interactions
│   │   ├── consistency.py # Performance variance metrics
│   │   └── encoding.py  # Target and categorical encoding
│   ├── eda.py           # Exploratory data analysis
│   ├── preprocessor.py  # Data cleaning and integration
│   ├── model.py         # XGBoost model wrapper
│   ├── predict.py       # Prediction logic
│   ├── train.py         # Development training (with validation)
│   ├── trainFinal.py    # Production training (all data)
│   └── split_data.py    # Temporal train/test splitting
├── frontend/            # Web interface (HTML, CSS, JS)
├── models/              # Trained model files (.pkl)
└── requirements.txt     # Python dependencies
```

## Quick Start (Local Development)

**Prerequisites:** Python 3.8+

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python src/backend/run_api.py

# Access web interface
# Open http://localhost:8000 in your browser
```

**Note:** For production use, visit the deployed version on Render.

## Usage

### Web Interface

1. Select two fighters from the dropdown menus
2. Click "Predict Fight"
3. View win probabilities and predicted winner

### API Endpoints

**List Fighters:**
```bash
GET /fighters
Response: {"fighters": ["Fighter Name 1", "Fighter Name 2", ...]}
```

**Predict Fight:**
```bash
POST /predict
Body: {"fighter1": "Fighter Name 1", "fighter2": "Fighter Name 2"}
Response: {
  "fighter1": "Fighter Name 1",
  "fighter2": "Fighter Name 2",
  "fighter1_win_probability": 0.6234,
  "fighter2_win_probability": 0.3766,
  "predicted_winner": "Fighter Name 1"
}
```

## License & Credits

Data sourced from publicly available UFC statistics. Model trained on historical fight data for educational purposes.

