from build_features import prepare_features
from train import train_model
from predict import make_prediction
import pandas as pd

'''
Script for comparing multiple models
'''

standardize = True
models = ['random_forest', 'gbdt', 'svc']
runs_per_model = 5
force_overwrite = True

# Data preprocessing
prepare_features('data/train.csv', 'data/train_processed.csv', force_overwrite)
prepare_features('data/val.csv', 'data/test_processed.csv', force_overwrite) # NOTE: we rename val to test as it makes more sense

# Train and test each model 5 times and report average scores
metrics_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1'])

for model in models:
    model_runs = []

    for run in range(runs_per_model):
        train_model('data/train_processed.csv', model=model, standardize=standardize)
        run_result = make_prediction('data/' + model + '.pkl', 'data/test_processed.csv', standardize=standardize)
        model_runs.append(run_result)

    model_runs = pd.DataFrame(model_runs, columns=['Accuracy', 'Precision', 'Recall', 'F1']).mean().rename(model)
    metrics_df = metrics_df.append(model_runs)

metrics_df = metrics_df.round(3)

print('\n Final metrics: \n', metrics_df)



