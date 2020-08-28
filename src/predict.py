import pandas as pd
import numpy as np
import pickle as pkl
import sklearn

def make_prediction(model, test_data, standardize):
    """Test your model: make predictions and calculate metrics.
    
    Args:
        model (str): directory of the model
        val_data (str): directory of the preprocessed test set
    """
    df = pd.read_csv(test_data, sep=";")

    target = df["Survived"].values
    feats = df.drop(columns=["Survived"]).values

    if standardize:
        scaler = pkl.load(open('data/scaler.pkl', 'rb'))
        feats = scaler.transform(feats)
        
    model_unpickle = open(model, 'rb')
    model = pkl.load(model_unpickle)

    predictions = model.predict(feats)

    print("Accuracy on test set is: ", np.round(sklearn.metrics.accuracy_score(target, predictions), 3))
    print("F1 on test set is: ", np.round(sklearn.metrics.f1_score(target, predictions), 3))


