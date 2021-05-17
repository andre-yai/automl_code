cd ./../../src/ 
export ConfigFile=./../models/Diabetes_Model.json
uvicorn main_modelServing:app --reload --host 0.0.0.0 --port 15400