# metrics_utils.py
from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

__all__ = ["scale_df"]

def scale_df(df: pd.DataFrame, method: str = "robust") -> pd.DataFrame:
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "zscore":
        scaler = StandardScaler(with_mean=True, with_std=True)
    else:
        scaler = RobustScaler()
    arr = scaler.fit_transform(df)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)
