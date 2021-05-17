# Classes 

1 - Input data
In this we: 
- get our data.
- select target. if classification and regression.
- perform factorization: 
    - Missing value imputation to eliminate nulls in the training dataset.
    - Categorical encoding to convert categorical features to numeric indicators.
    - Dropping high-cardinality features, such as record IDs.
    - Feature engineering (for example, deriving individual date parts from DateTime features)
    - Others...
- separate in train and test percentage - if test data not provided


2 - Model Trainning
- task type: classification, regression
- blocked algorithms: [] 
- perform algorithms:
    - list
    - model
- exit criteria: trainning time, metric score threshold
- cross_validation: method
- generate pickle file
- Primary metric : this can be in 2 

3 - GraphMetrics
- Feature importance
- Confusion matrix
- accuracy table

4 - Logs
- log metrics for later avaliation

5 - CompareModels
- Comparare performace of latest model with new generated Model. If there is not significant Data Drift.


in a machine learning lifecycle we have: collecting data -> data processing/cleaning -> dividing to trainning and test -> costructining model -> model evaluation -> generating new model and compare to old one.

main.py -> importDatabase.py -> model_training -> graphMetrics.py -> compareModels.py

Places to log: data statistics, model performace, model versioning 

entrada: 
{
    dataTypeProvider: '',
    dataName: '',
    dataLocation: '',
    targetVariable: '',
    primaryMetric: '',
    processingVariableMethods: ['']
    modelType: '',
    customModel: '',
    modelList: [],
    previousModel: ''
    overFittingMethod: ''
} 