import numpy as np
import cv2

from rich.console import Console
from rich.table import Table

from pathlib import Path

import typer
from typing_extensions import Annotated


class ArucoDetector:
    camera_matrix = np.array([
        [1698.75, 0.0, 1115.55],
        [0.0, 1695.98, 751.98],
        [0.0, 0.0, 1.0],
    ])

    dist_coeffs = np.array([-0.00670872, -0.1481124, -0.00250596, 0.00299921, -1.68711031])

    marker_length = 0.02  # meters

    def find_tags(self, frame: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Detect ArUco tags in ``frame`` and return ``[(id, rvec, tvec), ...]``.

        Returns an empty list if no tags are detected.
        """
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _rejected = detector.detectMarkers(frame)

        if ids is None or len(ids) == 0:
            return []

        object_points = np.array([
            [-self.marker_length / 2, +self.marker_length / 2, 0],
            [+self.marker_length / 2, +self.marker_length / 2, 0],
            [+self.marker_length / 2, -self.marker_length / 2, 0],
            [-self.marker_length / 2, -self.marker_length / 2, 0],
        ])

        detections: list[tuple[int, np.ndarray, np.ndarray]] = []
        for i in range(len(ids)):
            _, rvec, tvec = cv2.solvePnP(
                object_points, corners[i], self.camera_matrix, self.dist_coeffs
            )
            detections.append((ids[i][0].item(), rvec[:, 0], tvec[:, 0]))

        return detections


def detect(image_path: Annotated[Path, typer.Argument(help="Path to the image to process")]):
    image = cv2.imread(str(image_path))
    detector = ArucoDetector()
    detections = detector.find_tags(image)

    # Sort detections by tag ID
    detections.sort(key=lambda x: x[0])

    # Print the results in a pretty rich table
    table = Table(title="ArUco Tag Detections")
    table.add_column("Tag ID", justify="center", style="cyan", no_wrap=True)
    table.add_column("Rotation Vector (rvec)", justify="center", style="magenta")
    table.add_column("Translation Vector (tvec)", justify="center", style="green")

    for marker_id, rvec, tvec in detections:
        rvec_str = np.array2string(rvec, precision=3)
        tvec_str = np.array2string(tvec, precision=3)
        table.add_row(str(marker_id), rvec_str, tvec_str)

    console = Console()
    console.print(table)
