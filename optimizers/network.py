import numpy as np
import tensorflow as tf
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


def split_sequences(sequences, ins, out):
    sequences = np.array(sequences)
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + ins
        out_end_ix = end_ix + out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def update_weight(model, demands, past, future):
    X, y = split_sequences(demands, past, future)
    model.fit(X, y, epochs=10, verbose=0)


def get_model(init_data, global_weights, past, future, threshold, use_saved=False):
        if use_saved:
            try:
                print("Loaded Model ...")
                model = tf.keras.models.load_model('../models/init.hdf5')
                return model
            except:
                pass
        else:
            print("Starting to make model ...")
            X, y = split_sequences(init_data, past, future)
            model = Sequential()
            model.add(LSTM(32, activation='relu', input_shape=(past, threshold)))
            model.add(RepeatVector(future))
            # model.add(LSTM(256, activation='relu', return_sequences=True))
            # model.add(LSTM(256, activation='relu'))
            # model.add(RepeatVector(future))
            # model.add(LSTM(256, activation='relu', return_sequences=True))
            model.add(LSTM(64, activation='relu'))
            model.add(RepeatVector(future))
            model.add(LSTM(128, activation='relu', return_sequences=True))
            # model.add(TimeDistributed(Dense(2*threshold,)))
            model.add(TimeDistributed(Dense(threshold)))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='mse')
            print("Model is compiled, starting to train model..")
            model.set_weights(global_weights)
            model.fit(X, y, epochs=20, verbose=0)
            print("Model fitting complete...")
        #     model.save("./models/init.hdf5")
        # print("Model saved to ./models dir ...")
        return model

def predict_demand(model, demands):
    demands = np.array(demands)
    demands = demands.reshape((1, demands.shape[0], demands.shape[1]))
    predicted_demand = model.predict(demands, verbose=0)
    return predicted_demand[0][0]


def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = 423
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean/len(scaled_weight_list))
        
    return avg_grad