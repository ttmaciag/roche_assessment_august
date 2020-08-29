import pandas as pd
import numpy as np
import pickle as pkl
import sklearn

def make_prediction(model, test_data, standardize):
    """Test your model: make predictions and calculate metrics.
    
    Args:
        model (str): directory of the model
        val_data (str): directory of the preprocessed test set
        standardize (bool): if True, the features will be standardized with StandardScaler

    Returns:
        1D np.array with [accuracy, precision, recall, f1] on test set.
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

    accuracy = sklearn.metrics.accuracy_score(target, predictions)
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(target, predictions, average='binary')
    

    return [accuracy, precision, recall, f1]
    


