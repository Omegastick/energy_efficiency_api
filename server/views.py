"""
Views for the building energy efficiency prediction server.
"""

from flask import abort, jsonify, request
import numpy as np
import torch

from server import app


@app.route('/', methods=['POST'])
def main():
    """
    Pass the inputs to the stored model and return the output
    """
    inputs = request.json

    # Input validation
    if not isinstance(inputs, list):
        abort(400, "Input must be a valid JSON matrix")
    try:
        inputs = np.array(inputs, dtype=np.float32)
    except TypeError:
        abort(400, "Input must be a valid JSON matrix")
    except ValueError:
        abort(400, "Input must be a non-jagged matrix")
    if len(inputs.shape) > 2:
        abort(400, "Input must be a matrix of 2 or fewer dimensions")
    if inputs.shape[-1] != 8:
        abort(400, "Input's lowest dimenions must be 8")

    app.logger.info("Serving request for inputs:\n%s", inputs)

    model = app.config['MODEL']
    output = model(torch.Tensor(inputs).view((-1, 8)))

    return jsonify(output.squeeze().tolist())
