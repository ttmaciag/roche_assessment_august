from build_features import prepare_features
from train import train_model
from predict import make_prediction

standardize = True
# Data preprocessing
prepare_features('data/train.csv', 'data/train_processed.csv', True)
prepare_features('data/val.csv', 'data/test_processed.csv', True) # NOTE: we rename val to test as it makes more sense

# Training
train_model('data/train_processed.csv', 'data/model.pkl', standardize)

# Predictions
make_prediction('data/model.pkl', 'data/test_processed.csv', standardize)


