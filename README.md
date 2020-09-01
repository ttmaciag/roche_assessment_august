# Titanic Assessment Project

This repository contains multiple tasks on the Titanic dataset. To see the instructions for each task got to [this document](docs/TASKS.md).


## Branch-specific comments:
The directory `api/` contains two files:
* **app.py** that runs Flask on a local server
* **request.py** that allows to test the web API with a simple POST JSON request 

To run it:
1. Start the server: `$ python api/app.py`
2. Call the server in a new terminal instance: `$ python api/request.py`
3. You should recieve a list with binary classifications made by the model.

### Side-notes
This API is not too useful in its current form. I ran out of time to implement it properly. :(


