import pickle
import os

abs_path = os.path.abspath(__file__)
algo_path = os.path.abspath(os.path.join(abs_path,'..','..'))
print(algo_path)

def save_obj(obj, name):
    with open(algo_path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# Load a variable from file
def load_obj(name):
    with open(algo_path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
