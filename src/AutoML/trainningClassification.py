from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
import joblib
from datetime import datetime
import os

import numpy as np

class TrainningClassification():

    # def trainModel():
        ###

    def hasStringFeatures(self):
        condString = False
        data_types = self.data[self.dataset.feature_cols].dtypes.to_dict()
        for type in data_types.values():
           if(type == "object"):
                self.logging.warning("We have categorical data in our feature. Some models may not be applicable.")
                condString = True

        return condString

    def availableModels(self,hasStringFeatures = False,modelName = ''):
        modelList = [{"name": "LogisticRegression","func": self.LogisticRegressionModel, "acceptString": False}
        , {"name": "decisionTree","func": self.DecisionTreeClassiferModel, "acceptString": True}]
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
        modelSelection = self.availableModels(hasStringFeatures = False, modelName = modelName)[0]
        self.model = modelSelection["func"](params) 
        metrics_model = self.calculateMetrics(self.model,modelName)
        print(metrics_model)


    def trainModelList(self):
        hasStringFeatures = self.hasStringFeatures()
        metrics = []
        modelList = self.availableModels(hasStringFeatures = hasStringFeatures)
        for typeModel in modelList:
            name = typeModel["name"]
            model = typeModel["func"]()
            metrics_model = self.calculateMetrics(model,name)
            metrics.append(metrics_model)
            print(f"Name: {name} Evaluation: {metrics_model['evaluation']}")
        
        self.getBestModel(metrics)

    def calculateMetrics(self,model, name):
        metrics_model = {}
        metrics_model["name"] = name
        metrics_model["model"] = model
        metrics_model["evaluation"] = {}

        if('acurracy' in self.metrics_performace):
            acc = self.calculateAccurancy(model)
            metrics_model["evaluation"]["accuracy"] = acc

        if('roc' in self.metrics_performace):
            auc = self.calculateAuc(model)
            metrics_model["evaluation"]["roc"] = auc

        return metrics_model
    

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



        ###
    def LogisticRegressionModel(self,params={}): 
        reg =  0.1 
        params = {"C":1/ 0.1, "solver":"liblinear"}
        print('Training a logistic regression model with regularization rate of', reg)
        # run.log('Regularization Rate',  np.float(reg))
        model = LogisticRegression(**params).fit(self.label_training, self.target_training)
        return model

    def DecisionTreeClassiferModel(self): 
        print('Training a decision tree model')
        # run.log('Regularization Rate',  np.float(reg))
        model =  DecisionTreeClassifier().fit(self.label_training, self.target_training)
        return model

    def partitionTrainingAndTesting(self,partitionTrainingPerc):
        self.label_training, self.label_test , self.target_training, self.target_test = train_test_split(
            self.features, self.target, train_size=partitionTrainingPerc, random_state=0)
    
    def calculateAccurancy(self,model):
        # calculate accuracy
        y_hat = model.predict(self.label_test)
        acc = np.average(y_hat == self.target_test)
        #run.log('Accuracy', np.float(acc))
        return acc

    def calculateAuc(self,model):
        # calculate AUC
        y_scores = model.predict_proba(self.label_test)
        auc = roc_auc_score(self.target_test,y_scores[:,1])
        # run.log('AUC', np.float(auc))
        return auc

  
    def generateModelOutput(self,fileName):
        dirDate = datetime.today().strftime("%Y-%m-%d")
        print(dirDate)

        os.makedirs('outputs', exist_ok=True)
        os.makedirs(f'outputs/{dirDate}', exist_ok=True)
        fileOutput = "outputs/" + dirDate + "/" + fileName
        joblib.dump(value=self.model, filename=fileOutput)
        self.logging.info(f"Generating output file on : {fileOutput}")
    

    # def crossValidation():
        ###

    # def metricEvaluation():
        ###

        
    def __init__(self,logging,dataset,config):
        self.metrics_performace = config["metrics_performace"]
        self.primaryMetric = config["primaryMetric"]
        fileOutput = config["fileOutput"]

        self.customModel = config["customModel"]
        self.logging = logging

        self.dataset = dataset
        self.data = dataset.data
        self.features = dataset.features
        self.target = dataset.target

        self.partitionTrainingAndTesting(config["partitionTraining"])
        if(self.customModel == ''):
            self.trainModelList()
        else:
            self.trainCustomModel(self.customModel)
        self.generateModelOutput(fileOutput)
