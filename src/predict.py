import pandas as pd
import numpy as np
import pickle as pkl
import sklearn

def make_prediction(model, test_data, save_to_file=None):
    """Test the model: make predictions, calculate metrics and prints them.
    
    Args:
        model (str): directory of the model
        test_data (str): directory of the preprocessed test set
        save_to_file (str): path to save the results in .csv format; 
                            if None (default) the resulting array wil not be saved
                    
    """
    df = pd.read_csv(test_data, sep=";")

    target = df["Survived"].values
    feats = df.drop(columns=["Survived"]).values

        
    model_unpickle = open(model, 'rb')
    model = pkl.load(model_unpickle)

    predictions = model.predict(feats)

    print("Accuracy on test set is: ", np.round(sklearn.metrics.accuracy_score(target, predictions), 3))
    print("F1 on test set is: ", np.round(sklearn.metrics.f1_score(target, predictions), 3))

    if save_to_file is not None:
        df["Predicted"] = predictions
        df.to_csv(save_to_file, sep=";")


