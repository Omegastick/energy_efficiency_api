"""
Tests for view.py
"""
# pylint: disable=redefined-outer-name

from flask.testing import FlaskClient
import numpy as np
import pytest

from server import app


@pytest.fixture
def client():
    """
    Provides a test client for accessing the application.
    """
    with app.test_client() as test_client:
        yield test_client


def test_invalid_json_returns_400(client: FlaskClient):
    """
    When a request with invalid JSON is sent, a 400 error should be returned.
    """
    assert client.post('/', data="asd").status_code == 400


def test_inputs_of_incorrect_size_return_400(client: FlaskClient):
    """
    When a request where the lowest dimeniosn is larger or smaller than 8 is
    sent, a 400 error should be returned.
    """
    assert client.post('/', json=[0, 1, 2, 3, 4, 5, 6]).status_code == 400
    assert client.post(
        '/', json=[0, 1, 2, 3, 4, 5, 6, 7, 8]).status_code == 400
    assert client.post('/', json=[[1], [2]]).status_code == 400


def test_not_lists_return_400(client: FlaskClient):
    """
    When a request with any valid JSON that is not a list is sent, a 400 error
    should be returned.
    """
    assert client.post('/', json={"hello": "world"}).status_code == 400


def test_single_valid_inputs_return_correct_outputs(client: FlaskClient):
    """
    When a request with a valid JSON list of 8 numbers is sent, a JSON list of
    8 numbers should be returned.
    """
    response = client.post('/', json=[0, 1, 2, 3, 4, 5, 6, 7])
    assert np.array(response.json).shape == (2,)


def test_multiple_valid_inputs_return_correct_outputs(client: FlaskClient):
    """
    When a request with a valid JSON matrix of 2 lists of 8 numbers is sent, a
    JSON matrix of 2x2 numbers should be returned.
    """
    response = client.post('/', json=[[0, 1, 2, 3, 4, 5, 6, 7],
                                      [0, 1, 2, 3, 4, 5, 6, 7]])
    assert np.array(response.json).shape == (2, 2)
