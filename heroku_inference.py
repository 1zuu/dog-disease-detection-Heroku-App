import os
import json
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True

from variables import *


interpreter = tf.lite.Interpreter(model_path=model_converter)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def Inference(symtoms):
    symtoms = symtoms.astype(np.float32)
    input_shape = input_details[0]['shape']
    assert np.array_equal(input_shape, symtoms.shape), "Input tensor hasn't correct dimension"

    interpreter.set_tensor(input_details[0]['index'], symtoms)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def process_prediction_data(X, all_diseases, all_symtoms):
    symtoms = np.zeros(len(all_symtoms))
    for j in range(len(X)):
        x = X[j]
        if type(x) == str:
            x = x.lower()
            idx = all_symtoms.index(x)
            symtoms[idx] = 1
    return symtoms

def get_precautions(disease):
    disease = disease.strip().lower()
    df = pd.read_csv(precausion_path)
    df_cols = df.columns.values
    df[df_cols[0]] = df[df_cols[0]].str.lower()
    row = df.loc[df[df_cols[0]] == disease].values[0][1:]
    return row
    
def predict_precautions(symtoms, all_diseases, all_symtoms):
    symtoms = process_prediction_data(symtoms, all_diseases, all_symtoms)
    symtoms = symtoms.reshape(1,-1)
    P = Inference(symtoms)
    label = P.argmax(axis=-1)[0]
    disease = all_diseases[label]
    precausions = get_precautions(disease)
    precausions = {'precausion'+str(i): precausion for (i,precausion) in enumerate(precausions)}    
    return precausions, disease
