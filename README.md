# IPL Cricket Match Run Rate Prediction

## Project Overview
This project aims to predict the run rate per over in Indian Premier League (IPL) cricket matches. By leveraging historical ball-by-ball data, the goal is to develop a robust model that can forecast runs scored in upcoming overs, providing valuable insights for strategizing during live matches.

## Data Source
The primary data source is a comprehensive `deliveries.csv` file containing ball-by-ball information for numerous IPL matches.

## Methodology
The project follows a structured approach:

### 1. Data Loading & Initial Cleanup
- Loaded the `deliveries.csv` dataset.
- Dropped irrelevant columns such as `ball`, `batter`, `bowler`, `non_striker`, `batsman_runs`, `extra_runs`, `extras_type`, `player_dismissed`, `dismissal_kind`, and `fielder` to focus on match-level run rate prediction.

### 2. Data Standardization (Team Names)
- Identified and corrected inconsistencies in team names (e.g., 'Delhi Daredevils' to 'Delhi Capitals', 'Kings XI Punjab' to 'Punjab Kings').

### 3. Data Aggregation (Over-wise Conversion)
- Transformed ball-by-ball data into over-by-over summaries, aggregating `total_runs` and `is_wicket` (wickets).
- Grouped data by `match_id`, `inning`, and `over` to get `runs_per_over` and `wickets_per_over`.

### 4. Feature Engineering (Cumulative Features)
- Created `cumulative_runs` and `cumulative_wickets` features, representing the runs and wickets accumulated *before* the current over, essential for time-series modeling.

### 5. Final Encoding (Categorical to Numerical)
- Applied one-hot encoding to `batting_team` and `bowling_team` to convert categorical team names into a numerical format suitable for machine learning models.

### 6. Exploratory Data Analysis (EDA)
- **Histogram of Runs Per Over**: Revealed a non-normal distribution with clusters around low scores, average scores (4-8 runs), and a long tail for high-scoring overs.
- **Average Runs per Over Throughout an Inning (Line Plot)**: Clearly showed the characteristic 'U-shape' of a T20 inning, with high run rates in the Powerplay (Overs 0-5), a dip in Middle Overs (Overs 6-15), and a significant spike in Death Overs (Overs 16-19).
- **Autocorrelation Function (ACF) Plot**: Indicated significant positive autocorrelation at Lag 1 for runs per over, suggesting that the run rate of an over is influenced by the previous over's run rate (momentum).

### 7. Time Series Split
- Split the data into training (80% of matches) and testing (20% of matches) sets based on `match_id` to maintain the temporal integrity of the time-series data.
- Separated features (X) and target (y = `runs_per_over`) for both training and testing sets.

### 8. Model Development & Evaluation

#### ARIMA Model
- **Model**: An ARIMA(1,1,1) model was trained with external regressors (exog) from the feature set.
- **Performance (Test Set)**:
    - RMSE: 4.892 runs
    - MAE: 3.754 runs
- **Insights**: The ARIMA model provided a baseline, but its predictions were often conservative, struggling to capture the extreme high and low run rates.

#### XGBoost Model
- **Feature Engineering**: Introduced `runs_per_over_lag1` (lagged run rate from the previous over) as a crucial feature.
- **Model**: An XGBoost Regressor was trained on the enriched feature set.
- **Performance (Test Set)**:
    - RMSE: 4.711 runs
    - MAE: 3.641 runs
- **Feature Importance**: `wickets_per_over`, `over`, `cumulative_wickets`, and `runs_per_over_lag1` were identified as the most important features.
- **Adversarial Attack**: Implemented a basic adversarial attack to test model robustness by perturbing key features (`over`, `cumulative_runs`, `runs_per_over_lag1`, `wickets_per_over`, `cumulative_wickets`). The attack showed that small perturbations in input features could lead to noticeable changes in predictions, indicating areas for robustness improvement.

#### LightGBM Model
- **Model**: A LightGBM Regressor was trained using the same lagged feature set as XGBoost.
- **Performance (Test Set)**:
    - RMSE: 4.622 runs
    - MAE: 3.581 runs
- **Feature Importance**: Similar to XGBoost, `cumulative_runs`, `runs_per_over_lag1`, `over`, and `cumulative_wickets` were highly important.

## Conclusion

Comparing the three models:
- The **ARIMA** model served as a good baseline but was limited in capturing the non-linear dynamics of run rates.
- The **XGBoost** model significantly outperformed ARIMA, demonstrating better accuracy and the ability to leverage complex interactions between features. Its lower RMSE and MAE, along with its ability to follow the actual run rate fluctuations more closely, made it a strong contender.
- The **LightGBM** model further improved upon XGBoost's performance, achieving the lowest RMSE and MAE among the three. Visually, LightGBM's predictions tracked the actual run rate fluctuations most accurately, capturing more of the sudden changes.

**Overall, LightGBM is the preferred model for this non-linear run rate prediction task due to its superior performance in terms of error metrics and its visual representation of predictions.** The success of both tree-based models (XGBoost and LightGBM) highlights the importance of using machine learning algorithms capable of handling non-linear relationships and leveraging engineered time-series features like lagged run rates and cumulative statistics.

## Future Work
- Hyperparameter tuning for both XGBoost and LightGBM models.
- Incorporating additional features such as pitch conditions, player forms, and match context (e.g., playoff stages).
- Exploring more advanced time-series models or ensemble methods.
- Developing a more sophisticated adversarial attack strategy and implementing defense mechanisms to improve model robustness.
