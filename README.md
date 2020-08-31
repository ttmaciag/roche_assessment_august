# Titanic Assessment Project

This repository contains multiple tasks on the Titanic dataset. To see the instructions for each task got to [this document](docs/TASKS.md).


## Branch-specific comments:
To test the code run: `pytest src/tests`.
The tests are stored in 'src/tests/test_utils.py'.
NOTE: The tests should pass, but you might get a warning, which is bug in current numpy release.

### Side-notes
1. It is not a good practice to add csv files and (especially) models to git. It is better to use gitLFS for that or an internal sharing system. In this case this is not feasible or possible, so I only added the csv files, this means that the one test (test_predictions) will fail, because there is no test_model.pkl in the repo.

2. I admit that these tests do not cover the code well – they are just examples. Probably from the start I should have formatted the code in a more modular fashion and store the data and models differently. I wish I had the time to get back to that...

