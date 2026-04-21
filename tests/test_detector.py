from me235b.detector import ArucoDetector

import numpy as np
import cv2
import os

# Get assets path
ASSETS_PATH = os.path.join(os.path.dirname(__file__), "../assets")

# solvePnP translation (marker origin in camera frame), practice image fixture.
EXPECTED_TVEC_0 = np.array([0.157, -0.100, 0.542])
EXPECTED_TVEC_5 = np.array([0.0865, -0.0836, 0.489])


def test_detector():
    image_path = os.path.join(ASSETS_PATH, "aruco_detection_test_practice.png")
    image = cv2.imread(image_path)
    detector = ArucoDetector()
    detections = detector.find_tags(image)

    detections = {marker_id: (rvec, tvec) for marker_id, rvec, tvec in detections}

    assert 0 in detections, "Tag ID 0 not found in the image."
    assert 5 in detections, "Tag ID 5 not found in the image."

    assert np.isclose(detections[0][1], EXPECTED_TVEC_0, atol=1e-3).all()
    assert np.isclose(detections[5][1], EXPECTED_TVEC_5, atol=1e-3).all()
