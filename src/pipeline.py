from build_features import prepare_features
from train import train_model
from predict import make_prediction
'''
Script for a training and testing a model
'''
standardize = False
force_overwrite = True
model='random_forest'

# Data preprocessing
prepare_features('data/train.csv', 'data/train_processed.csv', force_overwrite)
prepare_features('data/val.csv', 'data/test_processed.csv', force_overwrite) # NOTE: we rename val to test as it makes more sense

# Training
train_model('data/train_processed.csv', model=model, standardize=standardize)

# Predictions and scores
scores = make_prediction('data/' + model + '.pkl', 'data/test_processed.csv', standardize=standardize)
scores = [ round(score, 3) for score in scores ]
print('\nScores on test set:')
print('Accuracy:', scores[0])
print('Precision:', scores[1])
print('Recall:', scores[2])
print('F1:', scores[3])

