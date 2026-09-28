import numpy as np

from predict import error_overlay


def test_error_overlay_marks_false_positive_and_false_negative_pixels():
    prediction = np.array([[0, 1], [1, 0]])
    target = np.array([[0, 0], [1, 1]])
    overlay = error_overlay(prediction, target)
    assert overlay[0, 1].tolist() == [255, 0, 0]
    assert overlay[1, 1].tolist() == [0, 0, 255]
