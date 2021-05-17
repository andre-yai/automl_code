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
        # =============== 1. Import Dataset ============================
        self.dataset = ImportDataset(logging,self.config["fileInput"])
        # print(dataset.data.dtypes)
        self.usedFeatureEng = []
        useDrop = self.dataset.featureDropCardinal()
        if(useDrop):
            self.usedFeatureEng.append([self.dataset.featureDropCardinal])
        self.dataset.setTargetValue(self.config["targetValue"])

    #================= 2. Graph Metrics ===========================
    def infoDatabase(self):
        self.metrics = GraphMetric(logging,self.dataset,self.config)
        self.missingFlag, self.featureStats, self.unbalancedFlag, self.targetStats = self.metrics.infoDataset()
        
    # =================== 3. Model Training ========================
    def trainingModel(self):
        self.training = TrainingModels(logging,self.dataset,self.config)

    #================== MetaData ==========================
    
    def readMetadata(self):
        try:
            with open(f'./../../models/{self.config["modelName"]}.json', 'r') as file:
                self.metadata = json.load(file)
        except:
            self.metadata = {}

    def readFileConfig(self,fileConfig):
        with open(fileConfig, 'r') as file:
            self.config = json.load(file)

    def writeMetadata(self):        
        output = {"file":self.training.fileOutput, "cols":self.dataset.feature_cols, "classNames": self.config["classNames"],
        "featureStats": [self.featureStats],  "targetStats": [self.targetStats]}
        
        with open(f'./../../models/{self.config["modelName"]}.json', 'w+') as file:
            json.dump(output, file)

    
# ['not-diabetic', 'diabetic']

    def __init__ (self,fileConfig):
        logging.info('::: Starting Application :::')

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

        if(self.metadata.keys() == []):
            logging.info(" New data!! No data drift!!")
        else:
            previousStats = self.metadata["featureStats"][0]
            changes, datadrifts = compareData(self.featureStats, previousStats)
            if(changes):
                print(f"Changes: {changes}, DataDrift:{datadrifts} ")

        logging.info('::: Training Model :::')
        self.trainingModel()
            
        self.writeMetadata()


AutoML(os.environ["ConfigFile"])