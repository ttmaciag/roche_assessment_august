# Titanic Assessment Project

This repository contains multiple tasks on the Titanic dataset. To see the instructions for each task got to [this document](docs/TASKS.md).


## Branch-specific comments:
To run the models in Docker:
1. Install Docker and make sure the Docker daemon is running.
2. In the ROOT of this repo run: `$ docker build -t titanic .`
3. Now you can either run  `$ docker run titanic` to automatically train and test the three models from previous tasks or run `docker run -it --entrypoint /bin/bash titanic` to enter the container via bash.
4. To exit the container run `exit`.


