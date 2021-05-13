import logging
from importDataset import ImportDataset
from trainningClassification import TrainningClassification
from graphMetrics import GraphMetric

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
        self.dataset.setTargetValue('Diabetic')

    #================= 2. Graph Metrics ===========================
    def infoDatabase(self):
        self.metrics = GraphMetric(logging,self.dataset)
        self.metrics.infoDataset()
        
    # =================== 3. Model Training ========================
    def trainningModel(self):
        if(self.config["modelType"] == 'Classification'):
            TrainningClassification(logging,self.dataset,self.config)

    def __init__ (self,config = ''):
        logging.info('Starting Application')
        self.config = {
                "fileInput": {"fileName":'./datasets/diabetes.csv', "importType":'File', "separator": ','},
                "modelType": 'Classification',
                "targetValue":'Diabetic',
                "partitionTraining": 0.7,
                "metrics_performace":['acurracy', 'roc'],
                "primaryMetric":'accuracy',
                "fileOutput": 'diabetes_model.pkl',
                "customModel": {'name': 'LogisticRegression', 'params': {"C":1/ 0.1, "solver":"liblinear"} }
        }
        
        self.getData()
        self.infoDatabase()
        self.trainningModel()


    # logging.debug('This is a debug message')
    # logging.warning('This is a warning message')
    # logging.error('This is an error message')
    # logging.critical('This is a critical message')
    # logging.info('END Application')

AutoML()