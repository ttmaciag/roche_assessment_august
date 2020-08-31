from preprocess import drop_features
from build_features import prepare_features
from train import train_model
from predict import make_prediction
from pathlib import Path

'''Script for running the whole ML pipeline
'''

data_path = Path('/Users/Tomek/git/roche_assessment_august/data')

# Data preprocessing
drop_features(data_path / 'train.csv', data_path / 'train_stage1.csv')
prepare_features(data_path / 'train_stage1.csv', data_path / 'train_stage2.csv', True)

drop_features(data_path / 'val.csv', data_path / 'test_stage1.csv') # we rename val to test, as it makes more sense
prepare_features(data_path / 'test_stage1.csv', data_path / 'test_stage2.csv', True)

# Training
train_model(data_path / 'train_stage2.csv', data_path / 'model.pkl')

# Predictions
make_prediction(data_path / 'model.pkl', data_path / 'test_stage2.csv', None)


