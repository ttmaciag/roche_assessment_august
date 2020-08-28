import pandas as pd
import numpy as np

def drop_features(input_file, output_file):
    """Drops NaNs and unnecessary features in .csv file and saves it.

    Args:
        input_file (.csv): input file directory.
        output_file (str): output file directory.
    """

    data = pd.read_csv(input_file, sep = ";")
    data.drop(columns=["Ticket", "Cabin", "PassengerId"], inplace=True)
    data = data.dropna()

    data.to_csv(output_file, sep=';', index=False)