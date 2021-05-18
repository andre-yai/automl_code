
from libs.model_serving.ModelInference import Prediction
import json

ConfigFile='./../models/Diabetes_Model.json'
ModelFolder = './../models'
pred = Prediction(ConfigFile,ModelFolder)

class TestClass:
    # ==========================================================
    # This function is responsable calling for testing prediction
    #===========================================================
    
    def test_pred(self):
        data = [[2,180,74,24,21,23.9091702,1.488172308,22]]
        result = json.loads(pred.predict(data))
        assert len(result) == 1
        assert result[0] == 'not-diabetic'
