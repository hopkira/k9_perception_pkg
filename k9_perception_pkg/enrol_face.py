import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from sensor_msgs.msg import CompressedImage
from k9_interfaces_pkg.msg import TrackedFaceArray


class FaceEnroller(Node):

    def __init__(
        self,
        identity,
        database_path,
        yunet_model,
        sface_model,
        samples,
    ):
        super().__init__("face_enroller")

        self.identity = identity
        self.samples_required = samples

        self.person_dir = (
            Path(database_path).expanduser() / identity
        )
        self.person_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.latest_frame = None
        self.latest_tracks = None

        self.embeddings_saved = 0
        self.last_saved_time = 0.0

        # Avoid storing almost identical consecutive frames.
        self.minimum_sample_interval = 0.6

        self.detector = cv2.FaceDetectorYN.create(
            yunet_model,
            "",
            (320, 320),
            0.8,
            0.3,
            50,
        )

        self.recogniser = cv2.FaceRecognizerSF.create(
            sface_model,
            "",
        )

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )

        self.create_subscription(
            CompressedImage,
            "/k9/camera/eye/image/compressed",
            self.image_callback,
            qos,
        )

        self.create_subscription(
            TrackedFaceArray,
            "/k9/perception/tracked_faces",
            self.tracks_callback,
            qos,
        )

        self.timer = self.create_timer(
            0.1,
            self.process,
        )

        self.get_logger().info(
            f"Enrolling '{identity}' "
            f"({samples} samples required)"
        )

    def image_callback(self, msg):
        encoded = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        )

        self.latest_frame = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

    def tracks_callback(self, msg):
        self.latest_tracks = msg

    @staticmethod
    def bbox_values(bbox):
        cx = float(bbox.center.position.x)
        cy = float(bbox.center.position.y)
        width = float(bbox.size_x)
        height = float(bbox.size_y)

        return cx, cy, width, height

    def crop_track(self, frame, bbox):
        cx, cy, width, height = self.bbox_values(bbox)

        # Give YuNet some surrounding context.
        margin = 0.30

        x1 = int(cx - width * (0.5 + margin))
        y1 = int(cy - height * (0.5 + margin))
        x2 = int(cx + width * (0.5 + margin))
        y2 = int(cy + height * (0.5 + margin))

        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def detect_face(self, crop):
        h, w = crop.shape[:2]

        if w < 80 or h < 80:
            return None

        # YuNet 2023mar is most reliable when fed sensible
        # dimensions. Use 320 x 320 here for enrolment.
        detector_image = cv2.resize(
            crop,
            (320, 320),
        )

        self.detector.setInputSize(
            (320, 320)
        )

        try:
            _, faces = self.detector.detect(
                detector_image
            )
        except cv2.error as exc:
            self.get_logger().warning(
                f"YuNet failed: {exc}"
            )
            return None

        if faces is None or len(faces) == 0:
            return None

        # Select the highest-confidence face.
        face = max(
            faces,
            key=lambda row: row[14],
        ).copy()

        scale_x = w / 320.0
        scale_y = h / 320.0

        face[0] *= scale_x
        face[1] *= scale_y
        face[2] *= scale_x
        face[3] *= scale_y

        # Landmark pairs occupy columns 4..13.
        for i in range(4, 14, 2):
            face[i] *= scale_x
            face[i + 1] *= scale_y

        return face

    def create_embedding(self, crop, face):
        try:
            aligned = self.recogniser.alignCrop(
                crop,
                face,
            )

            feature = self.recogniser.feature(
                aligned
            )

        except cv2.error as exc:
            self.get_logger().warning(
                f"SFace failed: {exc}"
            )
            return None

        embedding = np.asarray(
            feature,
            dtype=np.float32,
        ).flatten()

        # Explicit L2 normalisation makes later cosine
        # comparisons straightforward.
        norm = np.linalg.norm(embedding)

        if norm <= 0.0:
            return None

        return embedding / norm

    def process(self):

        if self.latest_frame is None:
            return

        if self.latest_tracks is None:
            return

        if len(self.latest_tracks.faces) != 1:
            if len(self.latest_tracks.faces) > 1:
                self.get_logger().warning(
                    "More than one face visible; "
                    "enrolment paused."
                )
            return

        now = time.monotonic()

        if (
            now - self.last_saved_time
            < self.minimum_sample_interval
        ):
            return

        tracked_face = self.latest_tracks.faces[0]

        crop = self.crop_track(
            self.latest_frame,
            tracked_face.bbox,
        )

        if crop is None:
            return

        face = self.detect_face(crop)

        if face is None:
            return

        embedding = self.create_embedding(
            crop,
            face,
        )

        if embedding is None:
            return

        self.embeddings_saved += 1
        self.last_saved_time = now

        filename = (
            self.person_dir
            / f"embedding_{self.embeddings_saved:03d}.npy"
        )

        np.save(
            filename,
            embedding,
        )

        self.get_logger().info(
            f"Saved sample "
            f"{self.embeddings_saved}/"
            f"{self.samples_required}: "
            f"{filename}"
        )

        if (
            self.embeddings_saved
            >= self.samples_required
        ):
            self.get_logger().info(
                f"Enrolment complete for "
                f"'{self.identity}'"
            )

            rclpy.shutdown()


def main(args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "identity",
        help="Name of person to enrol",
    )

    parser.add_argument(
        "--database",
        default="~/k9_data/faces",
    )

    parser.add_argument(
        "--yunet-model",
        default=(
            "~/k9_ws/src/k9_perception_pkg/"
            "models/"
            "face_detection_yunet_2023mar.onnx"
        ),
    )

    parser.add_argument(
        "--sface-model",
        default=(
            "~/k9_ws/src/k9_perception_pkg/"
            "models/"
            "face_recognition_sface_2021dec.onnx"
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10,
    )

    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = FaceEnroller(
        identity=parsed.identity,
        database_path=parsed.database,
        yunet_model=os.path.expanduser(
            parsed.yunet_model
        ),
        sface_model=os.path.expanduser(
            parsed.sface_model
        ),
        samples=parsed.samples,
    )

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            rclpy.shutdown()

        node.destroy_node()


if __name__ == "__main__":
    main()