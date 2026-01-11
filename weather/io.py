from pathlib import Path
import pandas as pd

def load_variable_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        format="mixed",
        dayfirst=True,
        errors="raise",
    )
    df = df.set_index("datetime")
    return df
