import pickle
#All helper functions used

import numpy as np
import tensorflow as tf  
import tensorflow.compat.v1.keras.backend as K 
tf.compat.v1.disable_eager_execution()
# import keras.backend as K

def load_encoders():
    file= open("model/lbl_Well.obj",'rb')
    enc_loaded_W = pickle.load(file)
    file.close()

    file= open("model/lbl_Form.obj",'rb')
    enc_loaded_F = pickle.load(file)
    file.close()

    file= open("model/lbl_Group.obj",'rb')
    enc_loaded_G = pickle.load(file)
    file.close()

    return enc_loaded_G, enc_loaded_F, enc_loaded_W 

def load_scaler():
    scaler = pickle.load(open('model/scaler.pkl','rb'))
    return scaler

def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]
 
    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))
 
    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row
 
    return X_aug
 
# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad
# Feature Augumentation   
def augment_features(input_features,depth, N_neig=1):
  X_aug = np.zeros((input_features.shape[0], input_features.shape[1]*(N_neig*2+2)))
  X_aug_win = augment_features_window(input_features, N_neig)
  X_aug_grad = augment_features_gradient(input_features, [depth])
  X_aug = np.concatenate((X_aug_win, X_aug_grad), axis=1)
  return X_aug


def f1_weighted(true, pred): 
    #Compute loss function F1_score
    predLabels = K.argmax(pred, axis=1)
    pred = K.one_hot(predLabels, 12) 

    ground_positives = K.sum(true, axis=0) + K.epsilon()       # = TP + FN
    pred_positives = K.sum(pred, axis=0) + K.epsilon()         # = TP + FP
    true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP
    
    precision = true_positives / pred_positives 
    recall = true_positives / ground_positives

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives) 
    weighted_f1 = K.sum(weighted_f1)

    
    return weighted_f1 #for metrics, return only 'weighted_f1'