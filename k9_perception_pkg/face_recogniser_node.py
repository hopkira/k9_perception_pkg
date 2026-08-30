import time
from dataclasses import dataclass
from typing import Dict, Optional

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

from k9_interfaces_pkg.msg import (
    TrackedFaceArray,
    RecognisedFace,
    RecognisedFaceArray,
)


@dataclass
class RecognitionState:
    identity: str = ""
    confidence: float = 0.0
    recognised: bool = False

    observations: int = 0
    last_attempt_monotonic: float = 0.0


class FaceRecogniserNode(Node):
    def __init__(self):
        super().__init__("face_recogniser")

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        self.declare_parameter(
            "image_topic",
            "/k9/camera/eye/image/compressed",
        )

        self.declare_parameter(
            "tracked_faces_topic",
            "/k9/perception/tracked_faces",
        )

        self.declare_parameter(
            "recognised_faces_topic",
            "/k9/perception/recognised_faces",
        )

        self.declare_parameter(
            "min_face_width",
            80,
        )

        self.declare_parameter(
            "min_face_height",
            80,
        )

        self.declare_parameter(
            "recognition_interval",
            0.5,
        )

        self.declare_parameter(
            "crop_margin",
            0.20,
        )

        self.image_topic = self.get_parameter(
            "image_topic"
        ).value

        self.tracked_faces_topic = self.get_parameter(
            "tracked_faces_topic"
        ).value

        self.recognised_faces_topic = self.get_parameter(
            "recognised_faces_topic"
        ).value

        self.min_face_width = int(
            self.get_parameter("min_face_width").value
        )

        self.min_face_height = int(
            self.get_parameter("min_face_height").value
        )

        self.recognition_interval = float(
            self.get_parameter("recognition_interval").value
        )

        self.crop_margin = float(
            self.get_parameter("crop_margin").value
        )

        # ------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )

        # ------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------

        self.image_subscription = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            qos,
        )

        self.tracks_subscription = self.create_subscription(
            TrackedFaceArray,
            self.tracked_faces_topic,
            self.tracks_callback,
            qos,
        )

        self.publisher = self.create_publisher(
            RecognisedFaceArray,
            self.recognised_faces_topic,
            qos,
        )

        # ------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_image_stamp = None

        self.track_states: Dict[int, RecognitionState] = {}

        self.frames_received = 0
        self.track_messages_received = 0
        self.recognition_attempts = 0

        self.get_logger().info(
            "Face recogniser started: "
            f"image={self.image_topic}, "
            f"tracks={self.tracked_faces_topic}, "
            f"output={self.recognised_faces_topic}"
        )

        self.get_logger().info(
            f"OpenCV version: {cv2.__version__}"
        )

    # ------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------

    def image_callback(self, msg: CompressedImage):
        self.frames_received += 1

        encoded = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            self.get_logger().warning(
                "Unable to decode camera JPEG"
            )
            return

        self.latest_frame = frame
        self.latest_image_stamp = msg.header.stamp

    # ------------------------------------------------------------
    # Bounding box helpers
    # ------------------------------------------------------------

    @staticmethod
    def bbox_values(bbox):
        cx = float(bbox.center.position.x)
        cy = float(bbox.center.position.y)

        width = float(bbox.size_x)
        height = float(bbox.size_y)

        return cx, cy, width, height

    def crop_face(self, frame, bbox):
        cx, cy, width, height = self.bbox_values(bbox)

        margin_x = width * self.crop_margin
        margin_y = height * self.crop_margin

        x1 = int(cx - width / 2.0 - margin_x)
        y1 = int(cy - height / 2.0 - margin_y)

        x2 = int(cx + width / 2.0 + margin_x)
        y2 = int(cy + height / 2.0 + margin_y)

        frame_height, frame_width = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    # ------------------------------------------------------------
    # Quality gate
    # ------------------------------------------------------------

    def face_is_large_enough(self, bbox):
        _, _, width, height = self.bbox_values(bbox)

        return (
            width >= self.min_face_width
            and height >= self.min_face_height
        )

    # ------------------------------------------------------------
    # Recognition backend placeholder
    # ------------------------------------------------------------

    def recognise_face(self, face_crop):
        """
        Recognition backend.

        This will shortly be replaced with:

            crop
              -> alignment
              -> ArcFace embedding
              -> cosine similarity
              -> identity

        For now return UNKNOWN so that we can validate the
        complete ROS pipeline independently of the model.
        """

        self.recognition_attempts += 1

        return "", 0.0, False

    # ------------------------------------------------------------
    # Track processing
    # ------------------------------------------------------------

    def tracks_callback(self, msg: TrackedFaceArray):
        self.track_messages_received += 1

        output = RecognisedFaceArray()
        output.header = msg.header

        if self.latest_frame is None:
            self.publisher.publish(output)
            return

        active_track_ids = set()

        now = time.monotonic()

        for tracked_face in msg.faces:

            track_id = int(tracked_face.track_id)
            active_track_ids.add(track_id)

            state = self.track_states.get(track_id)

            if state is None:
                state = RecognitionState()
                self.track_states[track_id] = state

                self.get_logger().info(
                    f"Recognition candidate track: {track_id}"
                )

            # ----------------------------------------------------
            # Only attempt recognition when:
            #
            # 1. We have not already recognised this track.
            # 2. Enough time has elapsed since the previous try.
            # 3. The face is large enough.
            # ----------------------------------------------------

            should_attempt = (
                not state.recognised
                and (
                    now - state.last_attempt_monotonic
                    >= self.recognition_interval
                )
                and self.face_is_large_enough(
                    tracked_face.bbox
                )
            )

            if should_attempt:

                crop = self.crop_face(
                    self.latest_frame,
                    tracked_face.bbox,
                )

                if crop is not None:

                    identity, confidence, recognised = (
                        self.recognise_face(crop)
                    )

                    state.observations += 1
                    state.last_attempt_monotonic = now

                    if recognised:
                        state.identity = identity
                        state.confidence = confidence
                        state.recognised = True

                        self.get_logger().info(
                            f"Track {track_id} recognised as "
                            f"{identity} "
                            f"(confidence={confidence:.3f})"
                        )

            # ----------------------------------------------------
            # Publish enriched track
            # ----------------------------------------------------

            recognised_face = RecognisedFace()

            recognised_face.track_id = track_id
            recognised_face.identity = state.identity
            recognised_face.recognition_confidence = (
                state.confidence
            )
            recognised_face.recognised = state.recognised

            recognised_face.bbox = tracked_face.bbox

            recognised_face.first_seen = (
                tracked_face.first_seen
            )

            recognised_face.last_seen = (
                tracked_face.last_seen
            )

            output.faces.append(recognised_face)

        # --------------------------------------------------------
        # Forget recognition state for tracks which no longer
        # exist.
        #
        # The tracker is therefore authoritative for track life.
        # --------------------------------------------------------

        expired_ids = [
            track_id
            for track_id in self.track_states.keys()
            if track_id not in active_track_ids
        ]

        for track_id in expired_ids:
            del self.track_states[track_id]

        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)

    node = FaceRecogniserNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()