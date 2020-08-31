import pandas as pd
from pathlib import Path

def prepare_features(input_file, output_file="data/train_bf.csv", force_write = False):
    """Builds features and saves to .csv file.

    Args:
        input_file (.csv): input file directory.
        output_file (str): output file directory (default=data/train_bf.csv).
        force_wrtie (bool): overwrite existing file if True (default=False)
    """
    df = pd.read_csv(input_file, sep=";")

    # create dummy features for Sex
    df["IsFemale"] = df["Sex"].replace(["male", "female"], [0, 1])
    df.drop(columns=["Sex"], inplace=True)

    # encode Embarked labels
    embark_labels = df["Embarked"].unique()
    df["Embarked"].replace(embark_labels, range(len(embark_labels)), inplace=True)

    # calculate family size and mark single passangers
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = 0
    df["IsAlone"].where(df["FamilySize"] == 1, 1, inplace=True)

    # safely save to csv
    output_file = Path(output_file)
    if output_file.is_file() and not force_write:
        print("File already exists. Change 'output_file' or allow 'force_write'.")

    else:
        df.to_csv(output_file, sep=';', index=False)