import numpy as np

from predict import error_overlay, limit_prediction_records


def test_error_overlay_marks_false_positive_and_false_negative_pixels():
    prediction = np.array([[0, 1], [1, 0]])
    target = np.array([[0, 0], [1, 1]])
    overlay = error_overlay(prediction, target)
    assert overlay[0, 1].tolist() == [255, 0, 0]
    assert overlay[1, 1].tolist() == [0, 0, 255]


def test_error_overlay_marks_wrong_foreground_class_separately():
    prediction = np.array([[2]])
    target = np.array([[1]])
    assert error_overlay(prediction, target)[0, 0].tolist() == [255, 255, 0]


def test_prediction_record_limit_keeps_prefix():
    records = [{"case_id": str(index)} for index in range(3)]
    assert limit_prediction_records(records, 2) == records[:2]
