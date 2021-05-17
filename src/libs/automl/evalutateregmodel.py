from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

class ModelRegEvaluation:

    def calculateMetrics(self,model, name):
        metrics_model = {}
        metrics_model["name"] = name
        metrics_model["model"] = model
        metrics_model["evaluation"] = {}

        if('mae' in self.metrics_performace):
            acc = self.calculateMae(model)
            metrics_model["evaluation"]["mae"] = acc

        if('msqe' in self.metrics_performace):
            auc = self.calculateMsqe(model)
            metrics_model["evaluation"]["msqe"] = auc
        
        return metrics_model 
 
    def calculateMae(self,model):
        y_scores = model.predict(self.feature_test)
        prec =  mean_absolute_error(self.target_test,y_scores, average='macro')
        return prec

    def calculateMsqe(self,model):
        y_scores = model.predict(self.feature_test)
        rec = rmean_squared_error(self.target_test,y_scores, average='macro')
        return rec


    def __init__(self,feature_test, target_test,metrics_performace) -> None:
        self.feature_test = feature_test
        self.target_test = target_test
        self.metrics_performace = metrics_performace