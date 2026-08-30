import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import rclpy
import threading

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

    candidate_identity: str = ""
    candidate_count: int = 0
    candidate_confidence: float = 0.0


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
            "face_database",
            "~/k9_data/faces",
        )

        self.declare_parameter(
            "yunet_model",
            (
                "~/k9_ws/src/k9_perception_pkg/"
                "models/face_detection_yunet_2023mar.onnx"
            ),
        )

        self.declare_parameter(
            "sface_model",
            (
                "~/k9_ws/src/k9_perception_pkg/"
                "models/face_recognition_sface_2021dec.onnx"
            ),
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
            0.30,
        )

        self.declare_parameter(
            "recognition_threshold",
            0.50,
        )

        self.declare_parameter(
            "recognition_margin",
            0.08,
        )

        self.declare_parameter(
            "top_matches",
            3,
        )

        self.declare_parameter(
            "confirmation_count",
            3,
        )

        # ------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------

        self.image_topic = self.get_parameter(
            "image_topic"
        ).value

        self.tracked_faces_topic = self.get_parameter(
            "tracked_faces_topic"
        ).value

        self.recognised_faces_topic = self.get_parameter(
            "recognised_faces_topic"
        ).value

        self.face_database = Path(
            self.get_parameter(
                "face_database"
            ).value
        ).expanduser()

        self.yunet_model = str(
            Path(
                self.get_parameter(
                    "yunet_model"
                ).value
            ).expanduser()
        )

        self.sface_model = str(
            Path(
                self.get_parameter(
                    "sface_model"
                ).value
            ).expanduser()
        )

        self.min_face_width = int(
            self.get_parameter(
                "min_face_width"
            ).value
        )

        self.min_face_height = int(
            self.get_parameter(
                "min_face_height"
            ).value
        )

        self.recognition_interval = float(
            self.get_parameter(
                "recognition_interval"
            ).value
        )

        self.crop_margin = float(
            self.get_parameter(
                "crop_margin"
            ).value
        )

        self.recognition_threshold = float(
            self.get_parameter(
                "recognition_threshold"
            ).value
        )

        self.recognition_margin = float(
            self.get_parameter(
                "recognition_margin"
            ).value
        )

        self.top_matches = int(
            self.get_parameter(
                "top_matches"
            ).value
        )

        self.confirmation_count = int(
            self.get_parameter(
                "confirmation_count"
            ).value
        )

        # ------------------------------------------------------------
        # Face models
        # ------------------------------------------------------------

        self.detector_width = 320
        self.detector_height = 320

        self.detector = cv2.FaceDetectorYN.create(
            self.yunet_model,
            "",
            (
                self.detector_width,
                self.detector_height,
            ),
            0.75,
            0.3,
            50,
        )

        self.recogniser = cv2.FaceRecognizerSF.create(
            self.sface_model,
            "",
        )

        # ------------------------------------------------------------
        # Load enrolled people
        # ------------------------------------------------------------

        self.face_database_embeddings = (
            self.load_face_database()
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

        self._image_lock = threading.Lock()
        self.latest_image_data = None
        self.latest_image_stamp = None

        self.track_states: Dict[
            int,
            RecognitionState,
        ] = {}

        self.frames_received = 0
        self.track_messages_received = 0
        self.recognition_attempts = 0

        # ------------------------------------------------------------
        # Startup diagnostics
        # ------------------------------------------------------------

        self.get_logger().info(
            "Face recogniser started: "
            f"image={self.image_topic}, "
            f"tracks={self.tracked_faces_topic}, "
            f"output={self.recognised_faces_topic}"
        )

        self.get_logger().info(
            f"OpenCV version: {cv2.__version__} "
            f"from {cv2.__file__}"
        )

        self.get_logger().info(
            "Recognition configuration: "
            f"threshold={self.recognition_threshold:.2f}, "
            f"margin={self.recognition_margin:.2f}, "
            f"top_matches={self.top_matches}, "
            f"confirmations={self.confirmation_count}"
        )

    # ------------------------------------------------------------
    # Database
    # ------------------------------------------------------------

    def get_latest_frame(self):

        with self._image_lock:

            if self.latest_image_data is None:
                return None

            image_data = (
                self.latest_image_data
            )

        encoded = np.frombuffer(
            image_data,
            dtype=np.uint8,
        )

        return cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

    def load_face_database(self):

        database = {}

        if not self.face_database.exists():
            self.get_logger().warning(
                f"Face database does not exist: "
                f"{self.face_database}"
            )
            return database

        for person_dir in sorted(
            self.face_database.iterdir()
        ):

            if not person_dir.is_dir():
                continue

            embeddings = []

            for filename in sorted(
                person_dir.glob("*.npy")
            ):

                try:
                    embedding = np.load(
                        filename
                    ).astype(
                        np.float32
                    ).flatten()

                except Exception as exc:
                    self.get_logger().warning(
                        f"Could not load {filename}: {exc}"
                    )
                    continue

                norm = np.linalg.norm(
                    embedding
                )

                if norm <= 0.0:
                    self.get_logger().warning(
                        f"Ignoring zero-length embedding: "
                        f"{filename}"
                    )
                    continue

                embedding /= norm

                embeddings.append(
                    embedding
                )

            if embeddings:

                database[
                    person_dir.name
                ] = embeddings

                self.get_logger().info(
                    f"Loaded {len(embeddings)} "
                    f"embeddings for "
                    f"'{person_dir.name}'"
                )

        self.get_logger().info(
            f"Face database contains "
            f"{len(database)} identities"
        )

        return database

    # ------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------

    def image_callback(
        self,
        msg: CompressedImage,
    ):

        self.frames_received += 1

        # Keep only the latest compressed JPEG.
        # Do not decode every camera frame.
        with self._image_lock:
            self.latest_image_data = bytes(
                msg.data
            )

            self.latest_image_stamp = (
                msg.header.stamp
            )

    # ------------------------------------------------------------
    # Bounding-box helpers
    # ------------------------------------------------------------

    @staticmethod
    def bbox_values(bbox):

        cx = float(
            bbox.center.position.x
        )

        cy = float(
            bbox.center.position.y
        )

        width = float(
            bbox.size_x
        )

        height = float(
            bbox.size_y
        )

        return (
            cx,
            cy,
            width,
            height,
        )

    def face_is_large_enough(
        self,
        bbox,
    ):

        _, _, width, height = (
            self.bbox_values(
                bbox
            )
        )

        return (
            width >= self.min_face_width
            and
            height >= self.min_face_height
        )

    # ------------------------------------------------------------
    # Crop the tracked face
    # ------------------------------------------------------------

    def crop_face(
        self,
        frame,
        bbox,
    ):

        (
            cx,
            cy,
            width,
            height,
        ) = self.bbox_values(
            bbox
        )

        margin_x = (
            width * self.crop_margin
        )

        margin_y = (
            height * self.crop_margin
        )

        x1 = int(
            cx
            - width / 2.0
            - margin_x
        )

        y1 = int(
            cy
            - height / 2.0
            - margin_y
        )

        x2 = int(
            cx
            + width / 2.0
            + margin_x
        )

        y2 = int(
            cy
            + height / 2.0
            + margin_y
        )

        frame_height, frame_width = (
            frame.shape[:2]
        )

        x1 = max(
            0,
            x1,
        )

        y1 = max(
            0,
            y1,
        )

        x2 = min(
            frame_width,
            x2,
        )

        y2 = min(
            frame_height,
            y2,
        )

        if (
            x2 <= x1
            or
            y2 <= y1
        ):
            return None

        return frame[
            y1:y2,
            x1:x2
        ]

    # ------------------------------------------------------------
    # Detect face landmarks within tracked crop
    # ------------------------------------------------------------

    def detect_face_in_crop(
        self,
        crop,
    ):

        crop_height, crop_width = (
            crop.shape[:2]
        )

        if (
            crop_width < 40
            or
            crop_height < 40
        ):
            return None

        detector_image = cv2.resize(
            crop,
            (
                self.detector_width,
                self.detector_height,
            ),
        )

        try:
            _, faces = self.detector.detect(
                detector_image
            )

        except cv2.error as exc:
            self.get_logger().warning(
                f"YuNet inference failed: {exc}"
            )
            return None

        if (
            faces is None
            or
            len(faces) == 0
        ):
            return None

        # Pick the highest-confidence face.
        face = max(
            faces,
            key=lambda row: row[14],
        ).copy()

        scale_x = (
            crop_width
            / self.detector_width
        )

        scale_y = (
            crop_height
            / self.detector_height
        )

        # Bounding box
        face[0] *= scale_x
        face[1] *= scale_y
        face[2] *= scale_x
        face[3] *= scale_y

        # Five facial landmarks
        for i in range(
            4,
            14,
            2,
        ):
            face[i] *= scale_x
            face[i + 1] *= scale_y

        return face

    # ------------------------------------------------------------
    # Generate SFace embedding
    # ------------------------------------------------------------

    def create_embedding(
        self,
        crop,
        face,
    ):

        try:

            aligned = (
                self.recogniser.alignCrop(
                    crop,
                    face,
                )
            )

            feature = (
                self.recogniser.feature(
                    aligned
                )
            )

        except cv2.error as exc:

            self.get_logger().warning(
                f"SFace inference failed: {exc}"
            )

            return None

        embedding = np.asarray(
            feature,
            dtype=np.float32,
        ).flatten()

        norm = np.linalg.norm(
            embedding
        )

        if norm <= 0.0:
            return None

        embedding /= norm

        return embedding

    # ------------------------------------------------------------
    # Match embedding against database
    # ------------------------------------------------------------

    def match_embedding(
        self,
        embedding,
    ):

        if not self.face_database_embeddings:
            return (
                "",
                0.0,
                False,
            )

        identity_scores = []

        for (
            identity,
            known_embeddings,
        ) in (
            self.face_database_embeddings.items()
        ):

            scores = [
                float(
                    np.dot(
                        embedding,
                        known_embedding,
                    )
                )
                for known_embedding
                in known_embeddings
            ]

            scores.sort(
                reverse=True
            )

            count = min(
                self.top_matches,
                len(scores),
            )

            identity_score = float(
                np.mean(
                    scores[:count]
                )
            )

            identity_scores.append(
                (
                    identity_score,
                    identity,
                )
            )

        identity_scores.sort(
            reverse=True
        )

        best_score, best_identity = (
            identity_scores[0]
        )

        if len(identity_scores) > 1:
            second_best_score = (
                identity_scores[1][0]
            )
        else:
            second_best_score = -1.0

        margin = (
            best_score
            - second_best_score
        )

        accepted = (
            best_score
            >= self.recognition_threshold
            and
            margin
            >= self.recognition_margin
        )

        self.get_logger().debug(
            f"Match: "
            f"{best_identity}={best_score:.3f}, "
            f"second={second_best_score:.3f}, "
            f"margin={margin:.3f}, "
            f"accepted={accepted}"
        )

        if not accepted:
            return (
                "",
                best_score,
                False,
            )

        return (
            best_identity,
            best_score,
            True,
        )

    # ------------------------------------------------------------
    # Complete recognition operation
    # ------------------------------------------------------------

    def recognise_face(
        self,
        face_crop,
    ):

        self.recognition_attempts += 1

        face = self.detect_face_in_crop(
            face_crop
        )

        if face is None:
            return (
                "",
                0.0,
                False,
            )

        embedding = self.create_embedding(
            face_crop,
            face,
        )

        if embedding is None:
            return (
                "",
                0.0,
                False,
            )

        return self.match_embedding(
            embedding
        )

    # ------------------------------------------------------------
    # Multi-observation confirmation
    # ------------------------------------------------------------

    def update_candidate(
        self,
        track_id,
        state,
        identity,
        confidence,
        recognised,
    ):

        if not recognised:

            # A failed observation doesn't immediately destroy
            # a candidate, but it doesn't advance it either.
            return

        if (
            identity
            == state.candidate_identity
        ):

            state.candidate_count += 1

            state.candidate_confidence = max(
                state.candidate_confidence,
                confidence,
            )

        else:

            state.candidate_identity = (
                identity
            )

            state.candidate_count = 1

            state.candidate_confidence = (
                confidence
            )

        self.get_logger().info(
            f"Track {track_id}: candidate "
            f"'{state.candidate_identity}' "
            f"{state.candidate_count}/"
            f"{self.confirmation_count} "
            f"(score={confidence:.3f})"
        )

        if (
            state.candidate_count
            >= self.confirmation_count
        ):

            state.identity = (
                state.candidate_identity
            )

            state.confidence = (
                state.candidate_confidence
            )

            state.recognised = True

            self.get_logger().info(
                f"Track {track_id} recognised as "
                f"'{state.identity}' "
                f"(confidence="
                f"{state.confidence:.3f})"
            )

    # ------------------------------------------------------------
    # Track callback
    # ------------------------------------------------------------

    def tracks_callback(
        self,
        msg: TrackedFaceArray,
    ):

        self.track_messages_received += 1

        output = RecognisedFaceArray()
        output.header = msg.header

        # If we have never received a camera image,
        # there is nothing available for recognition.
        with self._image_lock:
            have_image = (
                self.latest_image_data is not None
            )

        if not have_image:
            self.publisher.publish(
                output
            )
            return

        active_track_ids = set()

        now = time.monotonic()

        # Decode at most once during this tracker callback,
        # and only if a face actually needs recognition.
        frame = None

        for tracked_face in msg.faces:

            track_id = int(
                tracked_face.track_id
            )

            active_track_ids.add(
                track_id
            )

            state = self.track_states.get(
                track_id
            )

            if state is None:

                state = RecognitionState()

                self.track_states[
                    track_id
                ] = state

                self.get_logger().info(
                    f"Recognition candidate "
                    f"track: {track_id}"
                )

            # ----------------------------------------------------
            # Recognition attempt
            # ----------------------------------------------------

            should_attempt = (
                not state.recognised
                and
                (
                    now
                    - state.last_attempt_monotonic
                    >= self.recognition_interval
                )
                and
                self.face_is_large_enough(
                    tracked_face.bbox
                )
            )

            if should_attempt:
                # Only decode the JPEG when a recognition
                # attempt is genuinely required.

                state.last_attempt_monotonic = now

                if frame is None:
                    frame = self.get_latest_frame()

                if frame is not None:

                    crop = self.crop_face(
                        frame,
                        tracked_face.bbox,
                    )

                    if crop is not None:

                        (
                            identity,
                            confidence,
                            recognised,
                        ) = self.recognise_face(
                            crop
                        )

                        state.observations += 1

                        self.update_candidate(
                            track_id,
                            state,
                            identity,
                            confidence,
                            recognised,
                        )
            # ----------------------------------------------------
            # Publish enriched tracked face
            # ----------------------------------------------------

            recognised_face = (
                RecognisedFace()
            )

            recognised_face.track_id = (
                track_id
            )

            recognised_face.identity = (
                state.identity
            )

            recognised_face.recognition_confidence = (
                state.confidence
            )

            recognised_face.recognised = (
                state.recognised
            )

            recognised_face.bbox = (
                tracked_face.bbox
            )

            recognised_face.first_seen = (
                tracked_face.first_seen
            )

            recognised_face.last_seen = (
                tracked_face.last_seen
            )

            output.faces.append(
                recognised_face
            )

        # --------------------------------------------------------
        # Tracker owns track lifetime.
        # --------------------------------------------------------

        expired_ids = [
            track_id
            for track_id
            in self.track_states.keys()
            if track_id
            not in active_track_ids
        ]

        for track_id in expired_ids:

            state = self.track_states[
                track_id
            ]

            if state.recognised:
                self.get_logger().info(
                    f"Recognised face left: "
                    f"track={track_id}, "
                    f"identity='{state.identity}'"
                )

            del self.track_states[
                track_id
            ]

        self.publisher.publish(
            output
        )


def main(args=None):

    rclpy.init(args=args)

    node = FaceRecogniserNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()