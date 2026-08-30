import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from vision_msgs.msg import Detection2DArray
from k9_interfaces_pkg.msg import TrackedFace, TrackedFaceArray


class Track:
    def __init__(self, track_id, bbox, confidence, stamp):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.first_seen = stamp
        self.last_seen = stamp
        self.last_update_monotonic = time.monotonic()


class FaceTrackerNode(Node):
    def __init__(self):
        super().__init__('face_tracker')

        self.declare_parameter(
            'detections_topic',
            '/k9/perception/faces'
        )
        self.declare_parameter(
            'tracked_faces_topic',
            '/k9/perception/tracked_faces'
        )
        self.declare_parameter('max_track_age', 0.75)
        self.declare_parameter('max_center_distance', 180.0)
        self.declare_parameter('min_iou', 0.10)

        self.detections_topic = self.get_parameter(
            'detections_topic'
        ).value

        self.tracked_faces_topic = self.get_parameter(
            'tracked_faces_topic'
        ).value

        self.max_track_age = float(
            self.get_parameter('max_track_age').value
        )

        self.max_center_distance = float(
            self.get_parameter('max_center_distance').value
        )

        self.min_iou = float(
            self.get_parameter('min_iou').value
        )

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )

        self.subscription = self.create_subscription(
            Detection2DArray,
            self.detections_topic,
            self.detections_callback,
            qos,
        )

        self.publisher = self.create_publisher(
            TrackedFaceArray,
            self.tracked_faces_topic,
            qos,
        )

        self.tracks = {}
        self.next_track_id = 1

        self.get_logger().info(
            f'Face tracker started: '
            f'input={self.detections_topic}, '
            f'output={self.tracked_faces_topic}'
        )

    @staticmethod
    def bbox_values(bbox):
        cx = float(bbox.center.position.x)
        cy = float(bbox.center.position.y)
        width = float(bbox.size_x)
        height = float(bbox.size_y)

        x1 = cx - width / 2.0
        y1 = cy - height / 2.0
        x2 = cx + width / 2.0
        y2 = cy + height / 2.0

        return cx, cy, width, height, x1, y1, x2, y2

    def centre_distance(self, bbox_a, bbox_b):
        ax, ay, *_ = self.bbox_values(bbox_a)
        bx, by, *_ = self.bbox_values(bbox_b)

        return math.hypot(ax - bx, ay - by)

    def iou(self, bbox_a, bbox_b):
        _, _, _, _, ax1, ay1, ax2, ay2 = self.bbox_values(bbox_a)
        _, _, _, _, bx1, by1, bx2, by2 = self.bbox_values(bbox_b)

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection_width = max(0.0, ix2 - ix1)
        intersection_height = max(0.0, iy2 - iy1)

        intersection = intersection_width * intersection_height

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0.0:
            return 0.0

        return intersection / union

    def extract_confidence(self, detection):
        if not detection.results:
            return 0.0

        return float(
            detection.results[0].hypothesis.score
        )

    def find_best_track(self, bbox, available_track_ids):
        best_track_id = None
        best_score = -1.0

        for track_id in available_track_ids:
            track = self.tracks[track_id]

            distance = self.centre_distance(
                bbox,
                track.bbox
            )

            overlap = self.iou(
                bbox,
                track.bbox
            )

            if (
                distance > self.max_center_distance
                and overlap < self.min_iou
            ):
                continue

            distance_score = max(
                0.0,
                1.0 - distance / self.max_center_distance
            )

            score = (
                0.6 * overlap +
                0.4 * distance_score
            )

            if score > best_score:
                best_score = score
                best_track_id = track_id

        return best_track_id

    def expire_old_tracks(self):
        now = time.monotonic()

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if now - track.last_update_monotonic
            > self.max_track_age
        ]

        for track_id in expired:
            self.get_logger().debug(
                f'Expiring face track {track_id}'
            )
            del self.tracks[track_id]

    def detections_callback(self, msg):
        self.expire_old_tracks()

        available_track_ids = set(self.tracks.keys())

        current_tracks = []

        for detection in msg.detections:
            bbox = detection.bbox
            confidence = self.extract_confidence(detection)

            matched_track_id = self.find_best_track(
                bbox,
                available_track_ids
            )

            if matched_track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1

                track = Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    stamp=msg.header.stamp,
                )

                self.tracks[track_id] = track

                self.get_logger().info(
                    f'New face track: {track_id}'
                )

            else:
                track_id = matched_track_id
                track = self.tracks[track_id]

                track.bbox = bbox
                track.confidence = confidence
                track.last_seen = msg.header.stamp
                track.last_update_monotonic = time.monotonic()

                available_track_ids.remove(track_id)

            current_tracks.append(track)

        output = TrackedFaceArray()
        output.header = msg.header

        for track in current_tracks:
            tracked_face = TrackedFace()

            tracked_face.track_id = track.track_id
            tracked_face.bbox = track.bbox
            tracked_face.detection_confidence = track.confidence
            tracked_face.first_seen = track.first_seen
            tracked_face.last_seen = track.last_seen

            output.faces.append(tracked_face)

        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)

    node = FaceTrackerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()