cd ./../../src/ 
export ConfigFile=./../models/Diabetes_Model.json
uvicorn main_modelServing:app --reload 