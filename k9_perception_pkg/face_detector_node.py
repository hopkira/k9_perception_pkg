#!/usr/bin/env python3

import os
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
)

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32

from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class FaceDetectorNode(Node):
    """
    K9 eye-camera face detector.

    Input:
        /k9/camera/eye/image/compressed
            sensor_msgs/CompressedImage

    Outputs:
        /k9/perception/faces
            vision_msgs/Detection2DArray

        /k9/perception/face_count
            std_msgs/Int32

        /k9/perception/faces/debug/compressed
            sensor_msgs/CompressedImage, optional

    Detection is performed using OpenCV FaceDetectorYN / YuNet.
    """

    def __init__(self):
        super().__init__("face_detector")

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------

        self.declare_parameter(
            "image_topic",
            "/k9/camera/eye/image/compressed",
        )

        self.declare_parameter(
            "detections_topic",
            "/k9/perception/faces",
        )

        self.declare_parameter(
            "face_count_topic",
            "/k9/perception/face_count",
        )

        self.declare_parameter(
            "debug_topic",
            "/k9/perception/faces/debug/compressed",
        )

        self.declare_parameter(
            "model_path",
            "",
        )

        self.declare_parameter(
            "score_threshold",
            0.75,
        )

        self.declare_parameter(
            "nms_threshold",
            0.3,
        )

        self.declare_parameter(
            "top_k",
            50,
        )

        self.declare_parameter(
            "max_processing_rate",
            15.0,
        )

        self.declare_parameter(
            "publish_debug_image",
            True,
        )

        self.declare_parameter(
            "debug_jpeg_quality",
            80,
        )

        self.image_topic = (
            self.get_parameter("image_topic")
            .get_parameter_value()
            .string_value
        )

        self.detections_topic = (
            self.get_parameter("detections_topic")
            .get_parameter_value()
            .string_value
        )

        self.face_count_topic = (
            self.get_parameter("face_count_topic")
            .get_parameter_value()
            .string_value
        )

        self.debug_topic = (
            self.get_parameter("debug_topic")
            .get_parameter_value()
            .string_value
        )

        self.model_path = (
            self.get_parameter("model_path")
            .get_parameter_value()
            .string_value
        )

        self.score_threshold = (
            self.get_parameter("score_threshold")
            .get_parameter_value()
            .double_value
        )

        self.nms_threshold = (
            self.get_parameter("nms_threshold")
            .get_parameter_value()
            .double_value
        )

        self.top_k = (
            self.get_parameter("top_k")
            .get_parameter_value()
            .integer_value
        )

        self.max_processing_rate = (
            self.get_parameter("max_processing_rate")
            .get_parameter_value()
            .double_value
        )

        self.publish_debug_image = (
            self.get_parameter("publish_debug_image")
            .get_parameter_value()
            .bool_value
        )

        self.debug_jpeg_quality = (
            self.get_parameter("debug_jpeg_quality")
            .get_parameter_value()
            .integer_value
        )

        if not self.model_path:
            raise RuntimeError(
                "model_path parameter has not been set"
            )

        if not os.path.isfile(self.model_path):
            raise RuntimeError(
                f"YuNet model does not exist: {self.model_path}"
            )

        if self.max_processing_rate <= 0:
            raise RuntimeError(
                "max_processing_rate must be greater than zero"
            )

        self.minimum_processing_interval = (
            1.0 / self.max_processing_rate
        )

        self.last_processing_time = 0.0

        # --------------------------------------------------------------
        # YuNet detector
        #
        # The input size is changed to match the incoming image before
        # each detection. The initial 320x320 value is therefore only
        # required when constructing FaceDetectorYN.
        # --------------------------------------------------------------

        self.detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        )

        # --------------------------------------------------------------
        # ROS QoS
        #
        # The eye camera is BEST_EFFORT, so the subscriber must be
        # compatible with that QoS policy.
        # --------------------------------------------------------------

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )

        result_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.image_subscription = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self._image_callback,
            image_qos,
        )

        self.detection_publisher = self.create_publisher(
            Detection2DArray,
            self.detections_topic,
            result_qos,
        )

        self.face_count_publisher = self.create_publisher(
            Int32,
            self.face_count_topic,
            result_qos,
        )

        self.debug_publisher = self.create_publisher(
            CompressedImage,
            self.debug_topic,
            image_qos,
        )

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        self.frames_received = 0
        self.frames_processed = 0
        self.frames_skipped = 0
        self.faces_detected = 0
        self.decode_errors = 0

        self.stats_timer = self.create_timer(
            10.0,
            self._log_statistics,
        )

        self.get_logger().info(
            "Face detector started: "
            f"input={self.image_topic}, "
            f"max_rate={self.max_processing_rate:.1f} Hz, "
            f"threshold={self.score_threshold:.2f}, "
            f"model={self.model_path}"
        )

    def _image_callback(self, msg: CompressedImage):
        """
        Receive compressed JPEG, decode on the Jetson and run YuNet.

        We deliberately process at a lower rate than the camera stream.
        K9 still receives 30 camera frames/sec, while face detection
        initially runs at 15 Hz. This keeps perception latency low
        without doing unnecessary inference work.
        """

        self.frames_received += 1

        now = time.monotonic()

        if (
            now - self.last_processing_time
            < self.minimum_processing_interval
        ):
            self.frames_skipped += 1
            return

        self.last_processing_time = now

        # --------------------------------------------------------------
        # Decode JPEG
        # --------------------------------------------------------------

        jpeg_array = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            jpeg_array,
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            self.decode_errors += 1
            self.get_logger().warn(
                "Unable to decode eye-camera JPEG frame"
            )
            return

        self.frames_processed += 1

        height, width = frame.shape[:2]

        # YuNet needs to know the actual image dimensions.
        self.detector.setInputSize(
            (width, height)
        )

        # --------------------------------------------------------------
        # Face detection
        # --------------------------------------------------------------

        _, faces = self.detector.detect(frame)

        detection_array = Detection2DArray()
        detection_array.header = msg.header

        face_count = 0

        if faces is not None:

            for face in faces:

                face_count += 1
                self.faces_detected += 1

                # YuNet result layout:
                #
                #  0 x
                #  1 y
                #  2 width
                #  3 height
                #  4..13 five landmark x/y pairs
                # 14 confidence
                #
                x = float(face[0])
                y = float(face[1])
                w = float(face[2])
                h = float(face[3])
                confidence = float(face[14])

                detection = Detection2D()

                # Bounding box uses centre coordinates.
                detection.bbox.center.position.x = (
                    x + (w / 2.0)
                )

                detection.bbox.center.position.y = (
                    y + (h / 2.0)
                )

                detection.bbox.size_x = w
                detection.bbox.size_y = h

                hypothesis = ObjectHypothesisWithPose()

                hypothesis.hypothesis.class_id = "face"
                hypothesis.hypothesis.score = confidence

                detection.results.append(hypothesis)

                detection_array.detections.append(
                    detection
                )

                if self.publish_debug_image:
                    self._draw_face(
                        frame,
                        face,
                    )

        # --------------------------------------------------------------
        # Publish detections
        # --------------------------------------------------------------

        self.detection_publisher.publish(
            detection_array
        )

        count_msg = Int32()
        count_msg.data = face_count

        self.face_count_publisher.publish(
            count_msg
        )

        # --------------------------------------------------------------
        # Optional annotated diagnostic image
        # --------------------------------------------------------------

        if self.publish_debug_image:
            self._publish_debug_image(
                frame,
                msg,
            )

    def _draw_face(self, frame, face):
        """
        Draw bounding box and YuNet's five landmarks.

        YuNet supplies:
            right eye
            left eye
            nose
            right mouth corner
            left mouth corner
        """

        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])

        confidence = float(face[14])

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        landmark_indices = [
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
            (12, 13),
        ]

        for x_index, y_index in landmark_indices:

            landmark_x = int(face[x_index])
            landmark_y = int(face[y_index])

            cv2.circle(
                frame,
                (landmark_x, landmark_y),
                3,
                (0, 0, 255),
                -1,
            )

        cv2.putText(
            frame,
            f"face {confidence:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _publish_debug_image(
        self,
        frame,
        source_msg: CompressedImage,
    ):
        """
        Re-encode the annotated diagnostic frame.

        This is deliberately optional. The production perception path
        does not require the debug image.
        """

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                self.debug_jpeg_quality,
            ],
        )

        if not success:
            return

        debug_msg = CompressedImage()

        debug_msg.header = source_msg.header
        debug_msg.format = "jpeg"
        debug_msg.data = encoded.tobytes()

        self.debug_publisher.publish(
            debug_msg
        )

    def _log_statistics(self):

        self.get_logger().info(
            "Face detector statistics: "
            f"received={self.frames_received}, "
            f"processed={self.frames_processed}, "
            f"skipped={self.frames_skipped}, "
            f"faces={self.faces_detected}, "
            f"decode_errors={self.decode_errors}"
        )


def main(args=None):

    rclpy.init(args=args)

    node = None

    try:

        node = FaceDetectorNode()

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
