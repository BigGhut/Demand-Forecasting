import pandas as pd

from src.data.preprocessor import DataPreprocessor


def test_fill_missing_carries_static_attributes():
    pre = DataPreprocessor()
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-03"]),
        "store_nbr": [1, 1],
        "family": ["PRODUCE", "PRODUCE"],
        "sales": [5.0, 7.0],
        "onpromotion": [0.0, 1.0],
        "city": ["Quito", "Quito"],
        "state": ["Pichincha", "Pichincha"],
        "type": ["A", "A"],
        "cluster": [13, 13],
        "is_localized_holiday": [0, 1],
        "dcoilwtico": [40.0, 41.0],
        "transactions": [10.0, 12.0],
    })

    out = pre._fill_missing(df)
    gap = out.loc[out["date"] == "2023-01-02"].iloc[0]

    assert gap["sales"] == 0
    assert gap["city"] == "Quito"
    assert gap["state"] == "Pichincha"
    assert gap["type"] == "A"
    assert int(gap["cluster"]) == 13
    assert int(gap["is_localized_holiday"]) == 0
    assert not out[["city", "state", "type", "cluster", "is_localized_holiday"]].isna().any().any()
