from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np

class ModelClasEvaluation:

    def calculateMetrics(self,model, name):

        #======================================================================-
        # This function is responsable for orquestrating classification metrics.
        #=======================================================================

        metrics_model = {}
        metrics_model["name"] = name
        metrics_model["model"] = model
        metrics_model["evaluation"] = {}

        if('acurracy' in self.metrics_performace):
            acc = self.calculateAccuracy(model)
            metrics_model["evaluation"]["accuracy"] = acc

        if('roc' in self.metrics_performace):
            auc = self.calculateAuc(model)
            metrics_model["evaluation"]["roc"] = auc
        
        if('precision' in self.metrics_performace):
            precision = self.calculatePrecision(model)
            metrics_model["evaluation"]["precision"] = precision

        if('recall' in self.metrics_performace):
            recall = self.calculateRecall(model)
            metrics_model["evaluation"]["recall"] = recall

        if('f1_score' in self.metrics_performace):
            f1_score = self.calculateF1score(model)
            metrics_model["evaluation"]["F1_Score"] = f1_score

        return metrics_model 
    
    def calculateAccuracy(self,model):
   
        #================================================================
        # This function is responsable for calculating Accuracy
        #================================================================

        y_hat = model.predict(self.feature_test)
        acc = np.average(y_hat == self.target_test)
        return acc


    def calculateAuc(self,model):

        #================================================================
        # This function is responsable for calculating Roc-AUC
        #================================================================

        y_scores = model.predict_proba(self.feature_test)
        auc = roc_auc_score(self.target_test,y_scores[:,1])
        return auc

    def calculatePrecision(self,model):

        #================================================================
        # This function is responsable for calculating Precision
        #================================================================

        y_scores = model.predict(self.feature_test)
        prec = precision_score(self.target_test,y_scores, average='macro')
        return prec

    def calculateRecall(self,model):
        #================================================================
        # This function is responsable for calculating Recall
        #================================================================

        y_scores = model.predict(self.feature_test)
        rec = recall_score(self.target_test,y_scores, average='macro')
        return rec


    def calculateF1score(self,model):
        #================================================================
        # This function is responsable for calculating F1 score.
        #================================================================
        
        y_scores = model.predict(self.label_test)
        auc = f1_score(self.target_test,y_scores)
        return auc

    def __init__(self,feature_test, target_test,metrics_performace) -> None:
        self.feature_test = feature_test
        self.target_test = target_test
        self.metrics_performace = metrics_performace