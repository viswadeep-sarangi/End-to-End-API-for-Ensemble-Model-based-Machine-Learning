# End to End API for Ensemble Model based Machine Learning
##### @author: Viswadeep Sarangi

This project is an end-to-end machine learning (ML) model API based on the [UCI Heart Failure Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) with training, saving and predicting capabilities using 3 different ML algorithms, namely:
- Supprt Vector Machines (`svm`)
- Decision Trees (`decisiontree`)
- Neural Networks (`neuralnetwork`) 

The project is ready for deployment either on the local computer or on the cloud using services such as Amazon Web Services (AWS), Microsoft Azure or Google Cloud Platform (GCP)

# Table of Contents
-----------------
1. [Quick Start (with Docker)](#quick-start-with-docker)
2. [Quick Testing the API](#quick-testing-the-api)
    * [2.1 Basic Root API](#basic-root-api)
    * [2.2 Train API](#train-api)
    * [2.3 Saving Model API](#saving-model-api)
    * [2.4 Predict API](#predict-api)
3. [Dev Environment](#dev-environment)
4. [Running the Server Manually (without Docker)](#running-the-server-manually-without-docker)
5. [Motivation of the ML Architecture](#motivation-of-the-ml-architecture)

## 1. Quick Start (with Docker) <div id='quick-start-with-docker'/>
-----------------------------------------
- Clone the git repository
- Navigate into the cloned directory using the terminal
- Ensure [docker](https://www.docker.com/) is installed 
- Build the Docker image using the command 
```sh
docker build -t viswadeep_uci_heart_failure_prediction_image .
```
- Create a Docker container from the image using the command mentioned below. This would start the API service
```sh
docker run -d --name viswadeep_uci_heart_failure_prediction_container -p 8000:80 -e APP_MODULE="api:app" viswadeep_uci_heart_failure_prediction_image
```
- The API can now be accessed at [http://localhost:8000](http://localhost:8000)
- Visit [http://localhost:8000/docs](http://localhost:8000/docs) to see the list of API calls available

## 2. Quick Testing the API <div id='quick-testing-the-api'/>
-------------------------------
The [http://localhost:8000/docs](http://localhost:8000/docs) is a very intuitive way of understanding the nature of the API calls to be made to the service, including providing details about the `curl` commands that can be invoked to test it. 

Here's a few `curl` commands for quick reference:
#### 2.1 Basic root API <div id='basic-root-api'/>
-----------------
```sh
curl -X 'GET' \
  'http://localhost:8000/' \
  -H 'accept: application/json'
  ```
This returns a basic response body like:
```json
{
  "predictions": null,
  "error": "This is a test endpoint."
}
```

#### 2.2 Train API <div id='train-api'/>
----------
There are 3 different ML models that can be trained using this API call, by
- Specifying the name of the ML model to be trained `model_name`, which can be either of 
	* `svm` (Support Vector Machine)
	* `decisiontree` (Decision Tree)
	* `neuralnetwork` (Neural Network)
- Providing a .csv file containing the training data in the prescribed format
	* The prescription of the .csv file can be referred to at [UCI Heart Failure Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)
- Providing an optional parameter of saving the trained model to the local disk, `save_model_to_disk`
	* If set to `true`, this API call returns a unique name of the trained model, which can be used to download the model using the `/downloadmodel` API endpoint

The following `curl` command illustrates an example of making this API call
```sh
curl -X 'POST' \
  'http://localhost:8000/train?model_name=decisiontree&save_model_to_disk=true' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'csv_data_file=@heart_failure_clinical_records_dataset.csv;type=application/vnd.ms-excel'
  ```
In the above call, the `heart_failure_clinical_records_dataset.csv` is provided as the training .csv data file. This assumes that the terminal is currently in the directory which has the .csv file.

The resulting response provides the details of the trained model, along with its (optional) unique name
```json
{
  "accuracy": "75.55555555555556%",
  "unique_model_name": "e66466080b3b4378880db39a9e68aac0.sav",
  "details": "Model:DecisionTreeClassifier, Criterion:entropy, Accuracy:0.7555555555555555. Call the /downloadmodel API endpoint with the unique_model_name to save the model to disk",
  "error": null
}
```

The corresponding `GET` API call for the `/train` API endpoint, like so...
```sh
curl -X 'GET' \
  'http://localhost:8000/train' \
  -H 'accept: application/json'
  ```
... would exit grafully, by returning a HTTPException response code of `400   Error: Bad Request` with the following message:
```json
{
  "detail": "Send a POST request to this endpoint with 'model_name' (svm, decisiontree or neuralnetwork) and 'csv_data_file' data with a 'save_model_to_disk' option"
}
```

#### 2.3 Saving Model API <div id='saving-model-api'/>
--------------
A pre-trained model can be downloaded to save to the local disk with the following terminal command
```sh
curl -X 'POST' \
  'http://localhost:8000/downloadmodel?unique_model_name=e66466080b3b4378880db39a9e68aac0.sav' \
  -H 'accept: application/json' \
  -d ''
  ```
where the `unique_model_name` is the result of the `/train` POST API call
The resulting response can be saved on to any local disk

The corresponding `GET` call, like so...
```sh
curl -X 'GET' \
  'http://localhost:8000/downloadmodel' \
  -H 'accept: application/json'
  ```
  would result in a `400    Error: Bad Request` with the following message:
```json
{
  "detail": "Send a POST request to this endpoint with 'unique_model_name' of the previously trained model"
}
```

#### 2.4 Predict API <div id='predict-api'/>
---------
```sh
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": {
			"age":75,
			"anaemia":0,
			"creatinine_phosphokinase":582,
			"diabetes":0,
			"ejection_fraction":20,
			"high_blood_pressure":1,
			"platelets":265000,
			"serum_creatinine":1.9,
			"serum_sodium":130,
			"sex":1,
			"smoking":0,
			"time":4
  }
}'
```
The above command with the provided sample JSON file should return a response similar to:
```json
{
  "predictions": [
    {
      "svm": 1,
      "decisiontree": 1,
      "neuralnetwork": 0
    }
  ],
  "error": null
}
```
However, the corresponding `GET` API call, like so...
```sh
curl -X 'GET' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json'
  ```
...would exit gracefully with the following error message:
```json
{
  "predictions": null,
  "error": "Send a POST request to this endpoint with 'features' data."
}
```


## 3. Dev Environment <div id='dev-environment'/>
-----------------
The development was done using the  [PyCharm](https://www.jetbrains.com/pycharm/) IDE, using [Python 3.7](https://www.python.org/downloads/release/python-370/) as the language of choice.
The [requirements.txt](https://github.com/viswadeep-sarangi/uci-heart-failure-ensemble-models-api/blob/main/requirements.txt) file in this repository includes all the Python packages used, along with their versions. 

For quick reference, the following packages were used for development:
| Package Name | Version |
| ------ | ------ |
| scikit-learn | 0.24.1 |
| pandas | 1.2.3 |
| numpy | 1.20.1 |
| fastapi | 0.63.0 |
| pydantic | 1.8.1 |
| uvicorn | 0.13.4 |
| python-multipart | 0.0.5 |
|jsonpickle | 2.0.0 |
| aiofiles | 0.6.0 |
| pytest | 6.2.2 |

## 4. Running the server manually (without Docker) <div id = 'running-the-server-manually-without-docker'/>
----------------

To run the API service locally, without having to go through Docker, please execute the following instructions:
- Clone the git repository to a local directory
- Navigate to the cloned directory using the terminal
- Ensure `Python 3.7` and `pip` are installed on the local computer and are accessible thorugh the terminal
	* Ensure executing `python --version` and `pip --version` does not return an error
- Upgrade `pip` using the command `pip install --upgrade pip` in the terminal
- Once, the above steps are complete, install all the dependencies of the repository using the command:
```sh 
pip install -r requirements.txt 
```
- Once all the required packages are installed, the server is ready to be run
- Navigate to the `/app` directory in the clone repository, and execute the following command in the terminal:
```sh
python api.py
```
- This should run the server at [http://localhost:8000](http://localhost:8000)
- The above mentioned `curl` commands can now be used the same way 

## 5. Motivation of the ML architecture <div id='motivation-of-the-ml-architecture'/>
---------------
The API as well as the ML architecture was developed to ensure optimum modularity of code. Each ML model is housed in a separate class. However all the classes share the same structure for ease of compatibility with the rest of the project structure.

The ML model classes are defined in a way such that:
* Each model class can easily be manipulated without affecting the rest of the environment
* Multiple variations of the models can be tested internally and only the best performing model will be saved and returned. For example:
    * Support Vector Machines (`svm`) 
        * Trained with different kernel functions i.e. `linear`, `rbf`, and only the best performing model is saved
    * Decision Trees (`decisiontree`)
        * Both `gini` impurity and information `entropy` functions are evaluated for best performance
    * Artificial Neural Networks (`neuralnetwork`). Two different aspects are permutated to achieve the optimum result
        * Activation Functions: `logistic`, `tanh`, `relu`
        * Solver: `sgd`, `adam`
* Additional ML models can be added with relative ease without compromising compatibility with the rest of the API structure


----------















