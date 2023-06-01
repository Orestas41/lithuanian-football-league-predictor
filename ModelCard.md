# Football Match Winner Predictor

## Model Details

- Developer: Orestas Dulinskas
- Model date: 01/06/2023
- Model Version: 1.0.0
- Model type: LinearRegression from scikit-learn

## Model Description

The Football Match Winner Predictor predicts which side is more likely to win the match and what is the probability of the win. The model is based on a Linear Regression model with the following parameters: fit_intercept=True, copy_X=True, normalize=True, positive=False. The model takes home and away teams as string inputs and outputs a float value representing the probability of the winning side.

## Model Training Data

The model has been trained on the results of the last 5 seasons (2018-2023) of Lithuania’s ‘Optibet A-Lyga’ football league. The training data consists of approximately 800 matches, including the date of the match, home and away teams, and the outcome (home team win, away team win, or a draw).

Pre-processing:

- Drop NaN
- Converting date into a timestamp
- Home and Away team names encoded with LabelEncoder from scikit-learn
- Created target column ('Winner) with the outcome of the match extracted from the match result
- Dropped results column

## Model Evaluation Metrics

The model's performance is assessed using Mean Absolute Error (MAE) and R-squared (R2) scores. MAE measures the average absolute difference between the predicted and actual outcomes, while R2 indicates the proportion of the variance in the target variable that is predictable. A lower MAE value and a higher R2 value signify better model performance. The model was tested on 160 matches.

Model performance measures:

- MAE = 0.37
- R2 = 0.12

## Model Limitations

While the model has the potential to improve over time, it currently has limitations. The lack of available data on injuries, bookings, and other factors could have enhanced its performance. Additionally, due to variations in the number of matches played by different teams over the last 5 seasons (e.g., due to relegation or joining the league later), some teams may receive more accurate predictions than others. Results could have improved the model performance, but since no other features were available, the model was largely overfitting during development and regularisation was not enough to make model perform as expected.

## Model Retraining

To enhance performance, the model will undergo weekly retraining. This process involves scraping the latest data from the football league, concatenating it with the existing data, and retraining the model. If the model demonstrates improved performance, it will be considered for production use.

## Ethical Considerations

There is no sensitive information. There is no use of the data to inform decisions about matters important to human well-being - such as health or safety.

## Model Contact Information

For any inquiries or feedback related to the Football Match Winner Predictor model, please contact: orestasdulinskas@gmail.com.
