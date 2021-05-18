import json 
import joblib
import numpy as np

class Prediction:

    # Called when a request is received
    def predict(self,raw_data):
        # ==========================================================
        # This function is responsable predictions
        #===========================================================

        data = np.array(raw_data)

        print(data)
        # Get a prediction from the model
        predictions = self.model.predict(data)
        print(predictions)
        # Get the corresponding classname for each prediction (0 or 1)

        classNames = self.metadata["classNames"]
        predicted_classes = []
        for prediction in predictions:
            predicted_classes.append(classNames[prediction])
        # Return the predictions as JSON
        return json.dumps(predicted_classes)
    
    def __init__(self,metadataLocation,ModelFolder) -> None:
        # ==========================================================
        # This function is reading metadata and making prediction
        #===========================================================
 
        with open(metadataLocation, 'r') as j:
            self.metadata = json.load(j)
            model_path = ModelFolder +'/'+ self.metadata['file']
            self.model = joblib.load(model_path)