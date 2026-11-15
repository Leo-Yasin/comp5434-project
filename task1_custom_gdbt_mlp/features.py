"""Data cleaning + feature engineering for Dataset 1 (weather classification)."""
import numpy as np
import pandas as pd

CATS = {"Cloud Cover": ["clear", "cloudy", "overcast", "partly cloudy"],
        "Season": ["Autumn", "Spring", "Summer", "Winter"],
        "Location": ["coastal", "inland", "mountain"]}
NUM = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)",
       "Atmospheric Pressure", "UV Index", "Visibility (km)"]


def build_features(df: pd.DataFrame, clean: bool = True) -> pd.DataFrame:
    """Return a purely numeric feature frame.

    Data analysis showed Humidity / Precipitation (%) are percentages but ~3% of
    rows exceed 100 (dirty/noisy values). We keep an 'is_over100' flag (the
    anomaly itself may carry signal) and clip the value to 100 when clean=True.
    Categorical columns are one-hot encoded.
    """
    X = df[NUM].astype(float).copy()
    X["hum_over100"] = (X["Humidity"] > 100).astype(float)
    X["prec_over100"] = (X["Precipitation (%)"] > 100).astype(float)
    if clean:
        X["Humidity"] = X["Humidity"].clip(upper=100)
        X["Precipitation (%)"] = X["Precipitation (%)"].clip(upper=100)
    # simple physically-motivated interactions
    X["hum_x_prec"] = X["Humidity"] * X["Precipitation (%)"] / 100.0
    X["temp_minus_freeze"] = X["Temperature"].clip(-30, 60)
    X["vis_x_hum"] = X["Visibility (km)"] * (100 - X["Humidity"]) / 100.0
    for col, levels in CATS.items():
        for lv in levels:
            X[f"{col}={lv}"] = (df[col].str.strip() == lv).astype(float)
    return X
