from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from libs.automl.evalutateclasmodel import ModelClasEvaluation
from libs.automl.evalutateregmodel import ModelRegEvaluation

import joblib
from datetime import datetime
import os

class TrainingModels():

    def hasStringFeatures(self):
        condString = False
        data_types = self.data[self.dataset.feature_cols].dtypes.to_dict()
        for type in data_types.values():
           if(type == "object"):
                self.logging.warning("We have categorical data in our feature. Some models may not be applicable.")
                condString = True

        return condString

    def availableClassificationBaseModels(self,hasStringFeatures = False,modelName = ''):
        
        modelList = [{"name": "LogisticRegression","func": LogisticRegression, "acceptString": False, "params":[{"C": 10.0, "solver": "liblinear"}]},
        {"name": "DecisionTreeClassifier","func": DecisionTreeClassifier, "acceptString": True, "params":[{}]},
        {"name": " RandomForestClassifier","func":  RandomForestClassifier, "acceptString": True, "params":[{}]}]

        executeModels = []
        for model in modelList:
            # if there is a restrition to models that accept String features
            if(hasStringFeatures == True and model["acceptString"] == True):
                executeModels.append(model)
            
            # if there is no retriction to model
            if(hasStringFeatures == False and modelName == ''):
                executeModels.append(model)

            # if there is a specific model. I am supposing that if is a specific model so it knows the constraints of using it.
            if(modelName != '' and model["name"] == modelName):
                executeModels.append(model)
        return executeModels


    def availableRegressionBaseModels(self,hasStringFeatures = False,modelName = ''):
            
            modelList = [{"name": "LinearRegression","func": LinearRegression, "acceptString": False, "params":[{}]},
            {"name": "DecisionTreeRegression","func": DecisionTreeRegressor, "acceptString": True, "params":[{}]},
            {"name": " RandomForestRegression","func":  RandomForestRegressor, "acceptString": True, "params":[{}]}]

            executeModels = []
            for model in modelList:
                # if there is a restrition to models that accept String features
                if(hasStringFeatures == True and model["acceptString"] == True):
                    executeModels.append(model)
                
                # if there is no retriction to model
                if(hasStringFeatures == False and modelName == ''):
                    executeModels.append(model)

                # if there is a specific model. I am supposing that if is a specific model so it knows the constraints of using it.
                if(modelName != '' and model["name"] == modelName):
                    executeModels.append(model)
            return executeModels

    def trainCustomModel(self,modelConfig):
        modelName = modelConfig['name']
        params = modelConfig['params']

        if(self.trainType == "Classification"):
            modelSelection = self.availableClassificationBaseModels(hasStringFeatures = False, modelName = modelName)[0]
        else:
            modelSelection = self.availableRegressionBaseModels(hasStringFeatures = False, modelName = modelName)[0]


        self.model = self.runModel(modelSelection["func"],params) 
        metrics_model = self.modelEvaluation.calculateMetrics(self.model,modelName)
        print(metrics_model)


    def trainModelList(self):
        hasStringFeatures = self.hasStringFeatures()
        metrics = []
        if(self.trainType == "Classification"):
            modelList = self.availableClassificationBaseModels(hasStringFeatures = hasStringFeatures)
        else:
            modelList = self.availableRegressionBaseModels(hasStringFeatures = hasStringFeatures)

        
        # TODO:Tuning Models
        
        for typeModel in modelList:
            name = typeModel["name"]
            params = typeModel["params"]
            for setting in params :
                model = self.runModel(typeModel["func"],setting)
                metrics_model = self.modelEvaluation.calculateMetrics(model,name)
                metrics.append(metrics_model)
                print(f"Name: {name} Evaluation: {metrics_model['evaluation']}")
        
        self.getBestModel(metrics)
    
    def getBestModel(self,metrics):
        bestModel = ""
        bestModelName = ""
        bestPerformace = 0
        for metric in metrics:
            metric_evaluation = metric["evaluation"]
            if self.primaryMetric in metric_evaluation:
                if(metric_evaluation[self.primaryMetric] > bestPerformace):
                    bestModel = metric["model"]
                    bestModelName = metric["name"]
                    bestPerformace = metric_evaluation[self.primaryMetric]
        print(f"Best Model: {bestModelName} {self.primaryMetric} : {bestPerformace}")
        self.model = bestModel


    def runModel(self,model_exec, params):
        return model_exec(**params).fit(self.feature_training, self.target_training)

    def partitionTrainingAndTesting(self,partitionTrainingPerc):
        self.feature_training, self.feature_test , self.target_training, self.target_test = train_test_split(
            self.features, self.target, train_size=partitionTrainingPerc, random_state=0)

  
    def generateModelOutput(self,fileName):
        dirDate = datetime.today().strftime("%Y-%m-%d")

        os.makedirs('./../models', exist_ok=True)
        os.makedirs(f'./../models/{self.modelName}', exist_ok=True)
        os.makedirs(f'./../models/{self.modelName}/{dirDate}', exist_ok=True)
        self.fileOutput = "./../models/"+ self.modelName + "/"+ dirDate + "/" + fileName
        joblib.dump(value=self.model, filename=self.fileOutput)
        self.logging.info(f"Generating output file on : {self.fileOutput}")
    
        
    def __init__(self,logging,dataset,config):
        self.metrics_performace = config["metrics_performace"]
        self.primaryMetric = config["primaryMetric"]
        fileOutput = config["fileOutput"]
        self.modelName = config["modelName"]
        self.customModel = config["customModel"]
        self.logging = logging

        self.trainType = config["modelType"] 

        self.dataset = dataset
        self.data = dataset.data
        self.features = dataset.features
        self.target = dataset.target
        

        self.partitionTrainingAndTesting(config["partitionTraining"])
        if(self.trainType == "Classification"):
            self.modelEvaluation = ModelClasEvaluation(self.feature_test, self.target_test,self.metrics_performace)
        else:
            self.modelEvaluation = ModelRegEvaluation(self.feature_test, self.target_test,self.metrics_performace)


        if(self.customModel == ''):
            self.trainModelList()
        else:
            self.trainCustomModel(self.customModel)
            
        self.generateModelOutput(fileOutput)
