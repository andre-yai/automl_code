
#=========================
# Main Class for auto ML
#======================

import logging
from libs.automl.importDataset import ImportDataset
from libs.automl.trainingModel import TrainingModels
from libs.automl.graphMetrics import GraphMetric
from libs.automl.compareData import compareData
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

class AutoML:
    def getData(self):
        
        # ==========================================================
        # This function is responsable calling for importing dataset
        #===========================================================

        self.dataset = ImportDataset(logging,self.config["fileInput"])
        # print(dataset.data.dtypes)
        self.usedFeatureEng = []
        useDrop = self.dataset.featureDropCardinal()
        if(useDrop):
            self.usedFeatureEng.append([self.dataset.featureDropCardinal])
        self.dataset.setTargetValue(self.config["targetValue"])

    def infoDatabase(self):
        #=============================================================
        # This function is responsable for getting dataset Metrics
        #==============================================================

        self.metrics = GraphMetric(logging,self.dataset,self.config)
        self.missingFlag, self.featureStats, self.unbalancedFlag, self.targetStats = self.metrics.infoDataset()
        
    def trainingModel(self):
        # =============================================================
        # This function is responsable for calling the training model.
        #===============================================================

        self.training = TrainingModels(logging,self.dataset,self.config,self.modelFolder)

    
    def readMetadata(self):
        #=======================================================
        # This function is responsable for reading metadata
        #========================================================
        try:
            metadataOutput = self.modelFolder+'/'+self.config["modelName"] +'.json'
            with open(metadataOutput, 'r') as file:
                self.metadata = json.load(file)
        except:
            self.metadata = {}

    def readFileConfig(self,fileConfig):
        #================================================================
        # This function is responsable for reading the config model file.
        #================================================================
        
        with open(fileConfig, 'r') as file:
            self.config = json.load(file)

    def writeMetadata(self):        
        #=======================================================
        # This function is resposable for writing Metadata
        #========================================================

        output = {"file":self.training.fileOutput, "cols":self.dataset.feature_cols, "classNames": self.config["classNames"],
        "featureStats": [self.featureStats],  "targetStats": [self.targetStats]}
        
        metadataOutput = self.modelFolder +'/'+ self.config["modelName"] +'.json'
        with open(metadataOutput, 'w+') as file:
            json.dump(output, file)

    def __init__ (self,fileConfig, modelFolder):
        #=======================================================
        # This function orchestrate main flow
        #========================================================

        logging.info('::: Starting Application :::')

        self.modelFolder = modelFolder

        self.readFileConfig(fileConfig)
        self.readMetadata()
            
        logging.info('::: Import Dataset :::')
        self.getData()

        logging.info('::: Info Dataset :::')
        self.infoDatabase()

        if(self.missingFlag > 0):
            logging.info("Treat Missing")
        
        if(self.unbalancedFlag == 1):
            logging.info("Treat Unbalanced Data")

        if(len(self.metadata.keys()) == 0):
            logging.info(" New data!! No data drift!!")
        else:
            previousStats = self.metadata["featureStats"][0]
            changes, datadrifts = compareData(self.featureStats, previousStats)
            if(changes):
                print(f"Changes: {changes}, DataDrift:{datadrifts} ")

        logging.info('::: Training Model :::')
        self.trainingModel()
        self.writeMetadata()


AutoML(os.environ["ConfigFile"], os.environ["ModelFolder"])