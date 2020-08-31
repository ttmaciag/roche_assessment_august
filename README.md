# Titanic Assessment Project

<br/>

This repository contains multiple soultions to tasks performed on the Titanic dataset which are a part of a recruitment proess to Roche.

To see the instructions for each task got to [this document](docs/TASKS.md).

<br/>

----------------
## TL;DR
This repository contains code for training, testing and comparing three ML models on the Titanic dataset. 
To run the code:
1. `pip install requirements.txt` or use Docker ([instructions](#Docker)).
2. `python src/pipline.py` to train and test a single model.
3. `python pipline_multi_model.py` to train, test and compare multiple models.
----------------

<br/>

## Models
Running `python pipline_multi_model.py` should give similar results on the Titanic test set:

| Model          | Accuracy | Precision | Recall |   F1  |
|----------------|----------|-----------|--------|-------|
| RandomForest   | 0.791    | 0.819     | 0.672  | 0.738 |
| GBDecisionTree | 0.800    | 0.861     | 0.649  | 0.739 |
| SVC            | 0.811    | 0.827     | 0.721  | 0.769 |

<br/>

Go [here](/docs/MODELS.md) to read on the interpretation of these scores.

<br/>

## Data-processing and model-development:
Inside `src/` you'll find:

* `build_features.py` for feature engineering and preprocessing.
* `train.py` for training ML models.
* `predict.py` for generating predictions on a test dataset.
* `pipline.py` (script) for running the above functionalities for a single model.
* `pipline_multi_model.py` (script) for running the above functionalities for a multiple models and comparing them.

<br/>

## Docker
To run the repository inside a Docker container:

1. Install Docker and make sure the Docker daemon is running.
2. In the ROOT of this repo run: `$ docker build -t titanic`.
3. Now you can either run `$ docker run titanic` to automatically train and test the three models or execute `docker run -it --entrypoint /bin/bash titanic` to enter the container via bash.
4. To exit the container execute `exit`.

<br/>

## Unit-tests
Run basic unit-tests with `pytest /src/tests`. (keep in mind that test_model.py is not stored in the repo due to its size)

<br/>

----------------

## To Do
1. Wrap pipline.py and pipline_multi_model.py in parsers, to run them from shell.
2. Add more unit tests
3. Add more models (SVM with non-linear kernel)


