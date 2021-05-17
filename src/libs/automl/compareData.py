
def compareData(infoPreviousData, infoNewData, meanChange = 0, stdChange = 0):
    print("============================== ** ==============")
    dataCols = infoPreviousData.keys()

    dataDrifts = {}
    changes = 0
    for col in dataCols:
        changed_mean = abs (infoPreviousData[col]['mean'] - infoNewData[col]['mean']) >  meanChange
        changed_std = abs ( infoPreviousData[col]['std'] - infoNewData[col]['std'])  >  stdChange
   
        if(changed_std or changed_mean):
            changes = changes + 1
        
        dataDrifts[col] = {'mean': infoPreviousData[col]['mean'] - infoNewData[col]['mean'], 'std': infoPreviousData[col]['std'] - infoNewData[col]['std']}

    return changes,dataDrifts



