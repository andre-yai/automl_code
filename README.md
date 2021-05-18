# Auto ML e Model Serving

Nome: André Kenji Yai

## Part 1. Auto ML e Model Serving


This is a project that is trying to emulate an AutoML and Model Serving.

In this project we cover diferent aspects to ML train and deployment lifecycle.


In Training we have:
- Importing dataset: Here we are importing from a csv file.
- Doing Feature Engineering:  Removing high cardinallity features. 
- Separate data: Separate data into feature and target values and into training and testing.
- Model: Selecting from different types of model that can be suitable for the task or to a custom model provided in config.
- Evaluation: Metrics  in classification (accuracy, Roc-Auc,precision, recall, f1_score) and Metrics in Regression (MAE, MSQRT).
- Generate model: Generate a pickle model.
- It would be nice to have a ML-Flow attach to it just to save metrics

In Model Serving we have:
- Take models from previous step.
- Instanciate a web server. And join the needed files to config.

To see TODOS see at todoApp.txt

File Organization
- models/ - models and metadatarepository.
- examples/ -  examples for the project.
- src/ - main applications code 
- libs/ - submodules
- automl
Here we have some modules:
importDataset - responsable for importing and perform some feature eng.
graphMetrics - responsable for providing information about our dataset.
compareData - responsable for comparing Data to see data drifts.
trainingDataset - responsable for training algorithms classification and regression.
evaluateclasmodel - responsable for evaluating classification models
evaluateregmodel - responsable for evaluating regression models.

- modelServing
ModelInfrence - responsible for getting the model and making predictions.

### How to run 
1. Create a conda enviroment using enviroment.yml.
2. Certify that the data that you need is in datasets
3. Go to examples/diabetes. There we have some executables:
- diabete_model.sh - generates a model according to a specific model according to config_diabete.json settings.
- diabete_model_geral.sh : generates a model to certain task not specifing the model. It tries different models and select the best.
- diabete_server.sh - runs Api server for predictions.

### Testing
1. run pytest mlServing_test.sh will test the result o module prediction.

### Swagger

http://0.0.0.0:15400/docs

as input in predict you can set [[2,180,74,24,21,23.9091702,1.488172308,22]]

### Docker 

Run command 
'''
    docker-compose up -d  
'''
to containize docker image for model serving.

Then 
'''
    docker run -d automl_code_core_api_ml    
'''
And test in http://localhost:8000

### Input 

In config_diabete.json code we have the following config.

'''json
{"modelName": "Diabetes_Model", "fileInput": {"fileName": "./../../datasets/diabetes.csv", "importType": "File", "separator": ","}, 
"targetValue": "Diabetic", "partitionTraining": 0.7, "modelType": "Classification",  "classNames": ["not-diabetic", "diabetic"],
"customModel": {"name": "LogisticRegression", "params": {"C": 10.0, "solver": "liblinear"}}, "metrics_performace": ["acurracy", "roc"], 
"primaryMetric": "accuracy", "fileOutput": "diabetes_model.pkl"} 
'''

modelName - name of model, experiment,
fileInput - name of dataset used
targetValue - target value
partitionTraining - partition in training and test
modeltype - model type. It is working with: classification and regression
classNames - target class names used in classification predict results
customModel - parameters for a specific model working with logistic regression, linear regression, decision trees and random forest. It accepts arguments. if customModel == "" then it will try with others models for the task.
metrics_performace - metrics used to compare models. 
primaryMetric - metric used to select best model.
fileOutput - name of pickle

### Outputs

In models/
- new model to be used. 
- metadata of application.

Next we instanciate run_server.sh that will run a web server in FastAPI to our application. Here metadata from the previous step will be used.


