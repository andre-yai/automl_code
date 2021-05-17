import json 
import joblib
import numpy as np

class Prediction:

    # Called when a request is received
    def predict(self,raw_data):
        # Get the input data as a numpy array
        data = np.array(json.loads(raw_data)['data'])

        # Get a prediction from the model
        predictions = self.model.predict(data)
        # Get the corresponding classname for each prediction (0 or 1)
        
        classNames = self.metadata["classNames"]
        predicted_classes = []
        for prediction in predictions:
            predicted_classes.append(classNames[prediction])
        # Return the predictions as JSON
        return json.dumps(predicted_classes)
    
    def __init__(self,metadataLocation) -> None:
        with open(metadataLocation, 'r') as j:
            self.metadata = json.load(j)
            model_path = self.metadata['file']
            self.model = joblib.load(model_path)

if __name__ == '__main__': 
    x_new = [[2,180,74,24,21,23.9091702,1.488172308,22]]
    print ('Patient: {}'.format(x_new[0]))

    # Convert the array to a serializable list in a JSON document
    input_json = json.dumps({"data": x_new})
    result = pred.predict(input_json)
    print (result)