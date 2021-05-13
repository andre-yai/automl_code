import numpy as np

class GraphMetric:

    def infoDataset(self):
        data_count = (len(self.data))
        # run.log('observations', row_count)
        self.logging.info('Analyzing {} rows of data'.format(data_count))


        print(" ============ Labels info ========")

        print("===================================")
        summary_stats = self.data[self.feature_cols].describe().to_dict()
        for col in summary_stats:
            keys = list(summary_stats[col].keys())
            values = list(summary_stats[col].values())
            for index in range(len(keys)):
                #run.log_row(col, stat=keys[index], value = values[index])
                self.logging.info(f"{col} , Key: {keys[index]} , value: {values[index]} ")
                if(keys[index] == "count" and int(values[index])/data_count < 1):
                        self.logging.warning(" ====== Missing data!! ====== ")
            print("===================================")
        
        # TODO: If model type trainning
        print(" ============ Group Target ========")

        summary_stats = self.data.groupby(self.target_col).count().reset_index().values
        for group in summary_stats:
            group_values = np.array([value for value in group[1:]]).mean()
            percentage_group = group_values/data_count
            self.logging.info(f"Group {group[0]} Value {group_values} Perc {percentage_group} ")
            if(percentage_group <= 0.2):
                self.logging.warning("====== May have unbalanced Data!! ===========")

        print("===================================")

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

    def __init__(self,logging,dataset):
        # this function saves file to location
        self.logging = logging
        self.setDataset(dataset)




