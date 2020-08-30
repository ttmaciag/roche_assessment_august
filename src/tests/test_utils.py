import pandas as pd
import os
import sys
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

def test_building_features():

    from build_features import prepare_features

    input_data_dir = 'src/tests/test_dataset.csv' 
    target_data_dir = 'src/tests/test_target.csv'
    processed_data_dir = 'src/tests/test_processed.csv'

    prepare_features(input_data_dir, processed_data_dir)
    processed_data = pd.read_csv(processed_data_dir, sep=';')
    os.remove(processed_data_dir)

    target_data = pd.read_csv(target_data_dir, sep=';')

    pd.testing.assert_frame_equal(processed_data, target_data) 


def test_predictions():

    from predict import make_prediction

    test_data_dir = 'src/tests/test_target.csv'
    expected = [0.32, 0.10526315789473684, 0.10526315789473684, 0.10526315789473684]
    out = make_prediction('src/tests/test_model.pkl', test_data_dir, False)

    assert out == expected



