"""
Utilities for configuring the application.
"""

import os

from logging.config import dictConfig
from typing import Optional

from flask import Flask
import torch

from model.model import Model


def create_app(name: str) -> Flask:
    """
    Create and setup the application.
    """
    app = Flask(name)

    # Load the model weights
    model_path = None
    try:
        model_path = os.environ['MODEL_PATH']
        app.logger.info("Loading model from %s", model_path)
    except KeyError:
        app.logger.warning("MODEL_PATH environment variable not set, not "
                           "loading model")

    app.config['MODEL'] = get_model(model_path)

    return app


def get_model(model_path: Optional[bool]) -> torch.nn.Module:
    # Get model to load from an environment variable
    model = Model()
    model.eval()
    if not model_path:
        return model

    model.load_state_dict(torch.load(model_path))
    return model


def setup_logging():
    """
    Perform basic logging configuration.
    """
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }},
        'root': {
            'level': 'INFO',
            'handlers': ['wsgi']
        }
    })
