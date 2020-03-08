"""
Tests for model.py
"""

import torch

from model.model import Model


def test_model_output_has_correct_size():
    """
    When multiple data points are given to the model, it should return an array
    of results.
    Single inputs are no longer supported (since introducing batch norm).
    """
    model_obj = Model()
    model_obj.eval()
    assert model_obj(torch.rand((10, 8))).shape == (10, 2)
