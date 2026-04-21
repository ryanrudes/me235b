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

    def find_tags(
        self,
        frame: np.ndarray,
        *,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Detect ArUco tags in ``frame`` and return ``[(id, rvec, tvec), ...]``.

        Returns an empty list if no tags are detected.

        ``camera_matrix`` / ``dist_coeffs`` default to the lab calibration on
        the class; pass explicit values when ``frame`` was produced with
        different intrinsics (e.g. pinhole synthetic views, scaled ``K``).
        """
        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_ARUCO_ORIGINAL
        )
        parameters = cv2.aruco.DetectorParameters()

        # Tighten parameters to reduce false detections
        #parameters.minMarkerPerimeterRate = 0.0001  # Increase for higher strictness
        #parameters.adaptiveThreshWinSizeMax = 100 # Increase for large markers
        #parameters.polygonalApproxAccuracyRate = 0.15 # Stricter accuracy
        #parameters.useAruco3Detection = True

        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _rejected = detector.detectMarkers(frame)

        if ids is None or len(ids) == 0:
            return []

        # Rows must match ``detectMarkers`` corner order: index 0 is the marker
        # bitmap top-left (see OpenCV ArucoDetector canonical warp). Marker frame
        # is x right, y down, z out of the pattern — same convention as
        # :meth:`me235b.sim.ViserRenderer._draw_tag_primitive` ``local_corners``.
        h = float(self.marker_length) / 2.0
        object_points = np.array(
            [
                [-h, -h, 0.0],
                [h, -h, 0.0],
                [h, h, 0.0],
                [-h, h, 0.0],
            ],
            dtype=np.float64,
        )

        if camera_matrix is None:
            K = self.camera_matrix
        else:
            K = np.asarray(camera_matrix, dtype=float)
        if dist_coeffs is None:
            D = self.dist_coeffs
        else:
            D = np.asarray(dist_coeffs, dtype=float).reshape(-1)

        detections: list[tuple[int, np.ndarray, np.ndarray]] = []
        for i in range(len(ids)):
            _, rvec, tvec = cv2.solvePnP(object_points, corners[i], K, D)
            detections.append((ids[i][0].item(), rvec[:, 0], tvec[:, 0]))

        return detections


def detect(image_path: Annotated[Path, typer.Argument(help="Path to the image to process")]):
    image = cv2.imread(str(image_path))

    detector = ArucoDetector()
    detections = detector.find_tags(image)

    # Keep only the detections matching markers 0, 1, 2, 3, 4, 5
    #detections = [d for d in detections if d[0] in [0, 1, 2, 3, 4, 5]]

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

    # Draw the detections on the image
    for marker_id, rvec, tvec in detections:
        cv2.drawFrameAxes(image, detector.camera_matrix, detector.dist_coeffs, rvec, tvec, 0.01)

    max_width = max(image.shape[1], image.shape[0])
    target_width = 1024
    scale = target_width / max_width
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    console = Console()
    for image in Path("../assets").glob("*.jpg"):
        if "sim7" in image.name:
            console.print(f"Processing {image}")
            detect(image)
        #break
