import numpy as np

class GraphMetric:

    def infoDataset(self):
        self.countLinesDataset()
        missingFlag,featureStats = self.featuresStats()
        unbalancedFlag, targetStats = self.targetStats()
        
        return missingFlag, featureStats, unbalancedFlag,targetStats

    def countLinesDataset(self):
        self.data_count = (len(self.data))
        # run.log('observations', row_count)
        self.logging.info('Analyzing {} rows of data'.format(self.data_count))


    def featuresStats(self):
        summary_stats = self.data[self.feature_cols].describe().to_dict()
        featuresStats = {}
        countMissing = 0
        for col in summary_stats:
            featuresStats[col] = {}
            keys = list(summary_stats[col].keys())
            values = list(summary_stats[col].values())
            for index in range(len(keys)):
                #run.log_row(col, stat=keys[index], value = values[index])
                self.logging.info(f"{col} , Key: {keys[index]} , value: {values[index]} ")
                if(keys[index] == "count" and int(values[index])/self.data_count < 1):
                        self.logging.warning(" ====== Missing data!! ====== ")
                        countMissing = countMissing + 1
                featuresStats[col][keys[index]] = values[index]
            
            print(" ==================================== ")
        
        return countMissing, featuresStats

    def targetStats(self):
        unbalancedDataFlag = 0
        targetStats = {}
        if(self.config["modelType"] == 'Classification'):
            print(" ============ Group Target ========")
            summary_stats = self.data.groupby(self.target_col).count().reset_index().values
            for group in summary_stats:
                group_values = np.array([value for value in group[1:]]).mean()
                percentage_group = group_values/self.data_count
                self.logging.info(f"Group {group[0]} Value {group_values} Percentual {percentage_group*100} ")
                
                targetStats[str(group[0])] = percentage_group*100

                if(percentage_group < 0.1):
                    self.logging.warning("====== May have unbalanced Data!! ===========")
                    unbalancedDataFlag = unbalancedDataFlag + 1 

            print("===================================")
        
        return unbalancedDataFlag,targetStats
        
        #for key in summary_stats[0].keys :
        #    self.logging.info(f" Class: {col}, {key} : {summary_stats[0][key]} ")



    #def saveToLocation(self,fileName):
        # this function saves file to location
    
    # def constructConfusionMatrix(self):
        # this function saves file to location

    #def constructFeatureImportance(self):
        # this function saves file to location

    #def constructAccuracyTable(self):
        # this function saves file to location

    def setDataset(self, dataset):
        self.data = dataset.data
        self.feature_cols = dataset.feature_cols
        self.target_col = dataset.target_col

    def __init__(self,logging,dataset,config):
        # this function saves file to location
        self.config = config
        self.logging = logging
        self.setDataset(dataset)

      
        





