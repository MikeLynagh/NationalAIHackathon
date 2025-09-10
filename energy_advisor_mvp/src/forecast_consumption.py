import pandas as pd
import numpy as np
from xgboost import XGBRegressor

class ForecastConsumption():
    """
    Forecast energy consumption to determine cost
    """

    @staticmethod
    def _get_recent_months_data(df_import, months=4):
        """
        Filter for the most recent months
        """
        last_date = df_import["ds"].max()
        first_date = last_date - pd.DateOffset(months=months)
        df_recent = df_import[df_import["ds"] > first_date].reset_index(drop=True)
        return df_recent

    @staticmethod
    def _format_datetime_data(df_import):
        df_import["ds"] = df_import['timestamp']
        df_import = df_import.sort_values("ds").reset_index(drop=True)
        return df_import

    @staticmethod
    def _add_lag_features(df_recent):
        """
        Feature engineering: create lag features and time-based features
        """
        df_recent["y"] = df_recent["import_kw"]
        for lag in range(1, 5):  # 4 lag features (can tune)
            df_recent[f"lag_{lag}"] = df_recent["y"].shift(lag)
        df_recent["hour"] = df_recent["ds"].dt.hour
        df_recent["minute"] = df_recent["ds"].dt.minute
        df_recent["dayofweek"] = df_recent["ds"].dt.dayofweek
        return df_recent

    @staticmethod
    def _impute_null(df_data):
        return df_data.dropna().reset_index(drop=True)

    @staticmethod
    def _train_model(df_training):
        feature_cols = [f"lag_{lag}" for lag in range(1, 5)] + ["hour", "minute", "dayofweek"]
        X_train, y_train = df_training[feature_cols], df_training["y"]
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def _forecast_future_consumption(model, df_training, forecast_days=30):
        # Forecast next month (30 days) at 30-min intervals
        n_periods = int((forecast_days * 24 * 60) / 30)
        last_known = df_training.iloc[-1]
        future_dates = pd.date_range(start=last_known["ds"] + pd.Timedelta(minutes=30), periods=n_periods, freq='30min')

        # Prepare future dataframe for recursive prediction
        future_df = pd.DataFrame({'timestamp': future_dates})
        preds = []
        lags = [last_known[f"lag_{i}"] for i in range(1, 5)]
        lags = [last_known["y"]] + lags[:-1]  # update lags for next step

        for date in future_dates:
            hour = date.hour
            minute = date.minute
            dayofweek = date.dayofweek
            features = lags + [hour, minute, dayofweek]
            pred = model.predict(np.array(features).reshape(1, -1))[0]
            preds.append(pred)
            # update lags for next step
            lags = [pred] + lags[:-1]

        future_df["import_kw"] = preds
        return future_df

    
    def run_forecast_modelling(self, df_energy_consumption):
        df_ds_formatted = self._format_datetime_data(df_energy_consumption)
        df_recent = self._get_recent_months_data(df_ds_formatted)
        df_features =self._add_lag_features(df_recent)
        df_training = self._impute_null(df_features)
        model = self._train_model(df_training)
        df_forecast = self._forecast_future_consumption(model, df_training)
        return df_forecast

