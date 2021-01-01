import os
import json
import pandas as pd
from flask import Flask
from flask import jsonify
from flask import request

from variables import *
from heroku_inference import predict_precautions

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    symtoms = eval(message['symtoms'])
    precausions, disease = predict_precautions(symtoms, all_diseases, all_symtoms)
    response = {
            'diseases': disease,
            'precausions': precausions
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host=host, port=port, threaded=False, use_reloader=False)