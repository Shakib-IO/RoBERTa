# Import libraries
from flask import Flask, request, jsonify
import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime

# Create Flask Application
app = Flask(__name__)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base") # Define the Tokenizer
session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx") # Initialize onnxruntine session

@app.route("/predict", methods = ["POST"]) # Define Route

# Predict Function 
def predict():
    input_ids = torch.tensor(
        tokenizer.encode(request.json[0], add_special_tokens = True)
    ).unsqueeze(0)

    if input_ids.requires_grad:
        numpy_func = input_ids.detach().cpu().numpy()
    else:
        numpy_func = input_ids.cpu().numpy()
    
    inputs = {session.get_inputs()[0].name: numpy_func}
    out = session.run(None, inputs)

    result = np.argmax(out)

    return jsonify({"positive": bool(result)})

# Call the main function with a port 5000
if __name__ == "__main__":
    app.run(host= "0.0.0.0", port= 8000, debug=True)