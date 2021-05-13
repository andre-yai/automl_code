
import pandas as pd


class ImportDataset:

    # get data from File
    def getDataFromFile(self,location,separator=';'):
        try:
            self.data = pd.read_csv(location,sep=separator)
        except:
            print("File Not Found")
    
  
    #def featureEngMissing(self):
        # Featuere Missing

    #def featureMissingFillMean(self):
    
    #def featureEncoding(self):
        # Featuere Encoding
    

    def featureDropCardinal(self):
        # Drop Cardinal
        useDrop = False
        num_columns = self.data.count()[0]
 
        for col in self.data.columns:
            unique_values = self.data[col].nunique()
            data_type = self.data[col].dtype
            uniqueless = unique_values/num_columns
            if(uniqueless >= 0.9 and data_type != 'float'):
                self.logging.info(f"Variable {col} and {data_type} is {uniqueless}.")
                self.data = self.data.drop([col],axis='columns')
                useDrop = True
        return useDrop

    
    def setTargetValue(self,targetCol):
        self.target_col = targetCol
        self.target = self.data[targetCol].values
        self.feature_cols = [col for col in self.data.columns if (col != targetCol)]
        self.features = self.data[self.feature_cols].values
    
    
    def __init__(self,logging,fileInputConfig):
        print(fileInputConfig)
        self.logging = logging
        location = fileInputConfig["fileName"]
        importType = fileInputConfig["importType"]
        separator = fileInputConfig["separator"]

        if(importType == 'File'):
            self.getDataFromFile(location,separator)

    


    
if __name__ == '__main__':
   dataset = ImportDataset('./datasets/diabetes.csv','File',',')
   # print(dataset.data.dtypes)
   dataset.featureDropCardinal()
   dataset.setTargetValue('Diabetic')
   # dataset.partitionTrainingAndTesting(0.7)
