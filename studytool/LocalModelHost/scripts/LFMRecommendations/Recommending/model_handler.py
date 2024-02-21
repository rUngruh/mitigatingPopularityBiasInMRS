import os
import pickle

from Models.RankALSminimal import RankALS #change if you don't want to use the minimal model

# set model paths
modelPath = "../Models"

def load_model(modelName='RankALSmin'): #change if you don't want to use the minimal model
    """
    load the model from the model path
    """
    
    model = RankALS()
    print('Loading model...')
    
    loadPath = os.path.join(modelPath, modelName + '.pkl')
    
    try:
        with open(loadPath, "rb") as f:
            model = pickle.load(f)
        print('Model loaded.')
        return model
    except FileNotFoundError:
        print('Model file not found.')
        return None
    


