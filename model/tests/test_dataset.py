"""
Tests for dataset.py.
"""
import pytest

import model.dataset


def test_error_if_no_dataset():
    """
    When a Dataset object is created but the data can't be found and `download`
    isn't set to True, an error should be raised.
    """
    with pytest.raises(RuntimeError):
        model.dataset.EnergyEfficiencyDataset('/asd', download=False)


def test_train_split_is_bigger_than_test():
    """
    When the dataset is downloaded and the length is checked, the length of the
    train split should be larger than the length of the test split.
    """
    train_dataset = model.dataset.EnergyEfficiencyDataset('/tmp', train=True)
    test_dataset = model.dataset.EnergyEfficiencyDataset('/tmp', train=False)

    assert len(train_dataset) > len(test_dataset)


def test_data_has_correct_dimensions():
    """
    When data is retrieved from the datset, it should come in two arrays. A
    features array of 8 length and a target array of 2.
    """
    dataset = model.dataset.EnergyEfficiencyDataset('/tmp', train=False)
    data = dataset[0]
    assert data[0].size == 8
    assert data[1].size == 2


def test_transform_is_applied():
    """
    When a transform is given to the dataset and data is retrieved from it,
    the data should be modified by the transform.
    """
    dataset = model.dataset.EnergyEfficiencyDataset('/tmp',
                                                    train=False,
                                                    transform=lambda _: 0)
    assert dataset[0][0] == 0
