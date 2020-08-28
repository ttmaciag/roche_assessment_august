import numpy as np
import pandas as pd
import sklearn
import pickle as pkl
from sklearn.preprocessing import StandardScaler


def train_model(train_dir, model_out_dir, standardize):
    """Train the default Random Forrest on train set, save it and calculate 
    accuracy on training and validation sets.
    
    Args:
        train_dir (str): directory of preprecessed training set
        val_dir (str): directory of preprecessed validation set
    """
    df = pd.read_csv(train_dir, sep=";")

    # Split to training and validation sets, optionaly standardize and save scaler.
    from sklearn.model_selection import train_test_split

    y = df["Survived"].values
    X = df.drop(columns=["Survived"]).values

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pkl.dump(scaler, open('data/scaler.pkl', 'wb'))
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    

    # Create a classifier and select scoring methods.
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10)

    # Fit full model and predict on both train and val.
    clf.fit(X_train, y_train)

    preds_train = clf.predict(X_train)
    preds_val = clf.predict(X_val)

    # Claculate metrics.
    metric_names = ["Accuracy", "F1"]
    train_results = []
    val_results = []

    train_results.append(sklearn.metrics.accuracy_score(y_train, preds_train))
    val_results.append(sklearn.metrics.accuracy_score(y_val, preds_val))

    train_results.append(sklearn.metrics.f1_score(y_train, preds_train))
    val_results.append(sklearn.metrics.f1_score(y_val, preds_val))

    
    # Save model
    model_pickle = open(model_out_dir, 'wb')
    pkl.dump(clf, model_pickle)
    model_pickle.close()

    # Return metrics.
    for i, metric_name in enumerate(metric_names):
        print("{} on training set is {}.".format((metric_name), np.round(train_results[i], 3)))
        print("{} on validation set is {}.".format(metric_name, np.round(val_results[i], 3)))
