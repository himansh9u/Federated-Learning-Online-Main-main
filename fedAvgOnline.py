import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
import random
import shutil
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
from loader.load_data import *
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from optimizers.network import *
from optimizers.constrained import constrained_solve
import numpy as np
import tensorflow as tf
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import backend as K

import sys
import hashlib


BUF_SIZE = 65536

sha256 = hashlib.sha256()

def env_hash():
    with open("./.env", 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

load_dotenv()

Q = int(os.getenv("Q_INIT"))
past = int(os.getenv("PAST"))
V_0 = int(os.getenv("V_0"))
future = int(os.getenv("FUTURE"))
alpha = float(os.getenv("ALPHA"))
NumSeq = int(os.getenv("NUM_SEQ"))
threshold = int(os.getenv("THRESHOLD"))
train_memory = int(os.getenv("TRAIN_MEMORY"))
use_saved = os.getenv("USE_SAVED")=="True"
run_others = os.getenv("RUN_OTHERS")=="True"
cost_constraint = int(os.getenv("COST_CONSTRAINT"))
time_limit = float('inf') if os.getenv("TIME_LIMIT")=='inf' else int(os.getenv("TIME_LIMIT"))
Q = int(os.getenv("Q_INIT"))
path_to_input = os.getenv("PATH_TO_INPUT")
number_of_clients = int(os.getenv("NUM_OF_CLIENTS"))

cache_constraint = int(alpha*threshold)


our_path = f"./Experiment_Clients/"
if os.path.exists(our_path):
    shutil.rmtree(our_path)
os.makedirs(our_path)
copyfile("./.env", our_path+"/.env")


data = pd.read_csv(path_to_input, sep = ' ')
data.columns = ['Timestamp', 'File_ID', "File_Size"]
DataLength = len(data)
def create_clients(dataset, num_clients, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                dataset 
        args: 
            dataset: provided training dataset
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
    '''
    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    #randomize the dataset
    dataset = dataset.sample(frac = 1)

    #shard data and place at each client
    size = len(dataset)//num_clients
    shards = [dataset[i:i + size] for i in range(0, size*num_clients, size)]
    for i in range(len(shards)):
        shards[i].sort_values("Timestamp", inplace=True)
        shards[i].reset_index(drop=True, inplace=True)
    
    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    path = f"./Federated_Learning"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    for client_idx in range(num_clients):
        f = open(f"./Federated_Learning/{client_names[client_idx]}.txt", "w")
        for i in range(len(shards[client_idx])):
            f.write(str(shards[client_idx]['Timestamp'][i]))
            f.write(" ")
            f.write(str(shards[client_idx]['File_ID'][i]))
            f.write(" ")
            f.write(str(shards[client_idx]['File_Size'][i]))
            f.write("\n")
        f.close()
    # return {client_names[i] : shards[i] for i in range(len(client_names))}

# dict = create_clients(data, number_of_clients, 'clients')
# print(dict['clients_1'])


create_clients(data, number_of_clients)


class localModelClass:
    def __init__(self, dataLength, client_data):
        self.dataLength = dataLength
        self.client_data = client_data
        self.X_t_1 = np.zeros((threshold,))
        self.init_indices = random.sample(range(threshold), cache_constraint)
        self.X_t_1[self.init_indices] = 1
        self.queue = []
        self.err = []
        self.objective = []
        self.fetching_cost = []
        self.cache_hit = []
        self.prev_demands = []
        self.best_maximum = []
        self.hit_rate = []
        self.download_rate = []
        self.local_model = None
    


class SimpleMLP:
    @staticmethod
    def get_model(past, threshold):
        model = Sequential()
        model.add(LSTM(32, activation='relu', input_shape=(past, threshold)))
        model.add(RepeatVector(future))
        model.add(LSTM(64, activation='relu'))
        model.add(RepeatVector(future))
        model.add(LSTM(128, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(threshold)))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='mse')
        return model
    
smlp_global = SimpleMLP()
global_model = smlp_global.get_model(past, threshold)


# gamma = np.random.normal(0, 1, (threshold,))

local_models = []
path = f"./Federated_Learning"
for client_path in tqdm(os.listdir(path)):
    file = os.path.join(path, client_path)
    client_data = pd.read_csv(file, sep = ' ')
    client_data.columns = ['Timestamp', 'File_ID', "File_Size"]
    DataLength = len(client_data)
    local_models.append(localModelClass(DataLength, client_data))
print(f"{number_of_clients} Local Models created....")

for i in tqdm(range(NumSeq)):
    global_weights = global_model.get_weights()
    scaled_local_weight_list = list()

    for client in local_models:
        
        V = V_0
        if os.getenv("USE_ROOT_V")=="True": V *= (i+1)**0.5
        next_dem, time = get_demands(i, time_limit, client.client_data, client.dataLength, NumSeq, threshold)
        X_t = np.zeros((threshold,))
        init_indices = random.sample(range(threshold), cache_constraint)
        X_t[init_indices] = 1
        
        if i==past+future:
            client.local_model = get_model(client.prev_demands, global_weights, past, future, threshold, use_saved)
            #set local model weight to the weight of the global model
            # print(model.summary())
        elif i>past+future:
            to_train = client.prev_demands[max(0, i-train_memory):]
            client.local_model.set_weights(global_weights)
            update_weight(client.local_model, to_train, past, future)
            
            pred = predict_demand(client.local_model, client.prev_demands[i-past:])
            pred = np.maximum(pred, np.zeros((pred.size,)))
            pred = np.round(pred)
            np.array(client.prev_demands).mean(axis=0)

            delta_t = get_delta()
            X_t, obj = constrained_solve(pred, cache_constraint, cost_constraint, client.X_t_1, delta_t, Q, V, threshold)
        
            client.objective.append(obj)
            Delta = delta_t*np.linalg.norm(X_t-client.X_t_1, ord=1)/2
            client.fetching_cost.append(Delta)
            
            e = np.linalg.norm(next_dem-pred, ord=2)/len(pred)
            client.err.append(e)
            actual_cache_hit = np.dot(next_dem, X_t)
            client.cache_hit.append(actual_cache_hit)
            
            indices = np.argsort(next_dem)[::-1][:cache_constraint]
            final = np.zeros((threshold,))
            final[indices] = 1
            
            
            best = np.dot(next_dem, final)
            client.best_maximum.append(best)
                    
            Q = max(Q + Delta - cost_constraint, 0)
            client.queue.append(Q)

            client.hit_rate.append(np.dot(X_t, next_dem)/np.sum(next_dem))
            client.download_rate.append(np.sum(np.logical_and(X_t==1, client.X_t_1==0))/np.sum(next_dem))
            

        client.X_t_1 = X_t
        client.prev_demands.append(next_dem)
        if i>=past+future:
            scaled_local_weight_list.append(client.local_model.get_weights())

    if i>=past+future: 
        #to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)
        
        #update global model 
        global_model.set_weights(average_weights)

for client_no in range(number_of_clients):
    client_paths = f"./Experiment_Clients/client_{client_no+1}/"
    try:
        os.makedirs(client_paths)
    except FileExistsError:
        pass

index = 0
for client in local_models:
    index+=1
    client_path = f"./Experiment_Clients/client_{index}/"
    plt.plot(ma(client.cache_hit))
    plt.title("Cache Hit vs Timeslot")
    plt.xlabel("Timeslot")
    plt.ylabel("Cache Hit")
    plt.savefig(client_path+"Cache_Hit.jpg")
    plt.clf()
    
    plt.plot(ma(client.err))
    plt.title("Mean Squared Test Error in Demand Prediction vs Timeslot")
    plt.xlabel("Timeslot")
    plt.ylabel("MSE")
    plt.savefig(client_path+ "NN-MSE.jpg")
    plt.clf()


    plt.plot(ma(client.queue))
    plt.title("Q vs Timeslot")
    plt.xlabel("Timeslot")
    plt.ylabel("Q")
    plt.savefig(client_path+ "Q.jpg")
    plt.clf()


    plt.plot(ma(client.objective))
    plt.title("Constrained Objective Function vs Timeslot")
    plt.xlabel("Timeslot")
    plt.ylabel("Objective Function")
    plt.savefig(client_path+"Obj.jpg")
    plt.clf()


    plt.plot(ma(client.fetching_cost))
    plt.title("Fetching Cost vs Timeslot")
    plt.axhline(y=cost_constraint, linewidth=2, label= 'Cost Constraint')
    plt.xlabel("Timeslot")
    plt.ylabel("Cost")
    # plt.legend(loc = 'upper left')
    plt.savefig(client_path+ "Cost.jpg")
    plt.clf()


    plt.plot(ma(client.cache_hit))
    plt.title("Cache Hit vs Timeslot")
    plt.xlabel("Timeslot")
    plt.ylabel("Cache Hit")
    plt.savefig(client_path+"Cache_Hit.jpg")
    plt.clf()
    
    plt.plot(ma(client.hit_rate))
    plt.title("Cache Hit Rate vs Timeslot")
    plt.xlabel("Timeslot")
    plt.ylabel("Cache Hit Rate")
    plt.savefig(client_path+"Cache_Hit_Rate.jpg")
    plt.clf()
    
    plt.plot(ma(client.download_rate))
    plt.title("Download Rate vs Timeslot")
    plt.xlabel("Timeslot")
    plt.ylabel("Download Rate")
    plt.savefig(client_path+"Download_Rate.jpg")
    plt.clf()
    
    pd.DataFrame(client.hit_rate).to_csv(client_path+'hit_rate.csv',index=False)
    pd.DataFrame(client.download_rate).to_csv(client_path+'download_rate.csv',index=False)


