#!/usr/bin/env python3
"""Command-line test client for K9's ROS face-enrolment API.

This no longer duplicates YuNet/SFace inference locally. The live
face_recogniser node owns capture, staging, metadata and database reload.
"""

import argparse
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from k9_interfaces_pkg.action import CaptureFace
from k9_interfaces_pkg.srv import CommitFaceEnrollment


class FaceEnrollerClient(Node):

    def __init__(self) -> None:
        super().__init__("face_enroller")

        self.capture_client = ActionClient(
            self,
            CaptureFace,
            "/face_recogniser/capture_face",
        )

        self.commit_client = self.create_client(
            CommitFaceEnrollment,
            "/face_recogniser/commit_enrolment",
        )

        self._last_feedback_state = ""

    def _feedback_callback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback

        if feedback.state != self._last_feedback_state:
            self._last_feedback_state = feedback.state
            self.get_logger().info(
                f"Capture state: {feedback.state} "
                f"({feedback.samples_collected}/"
                f"{feedback.samples_required})"
            )

    def capture(
        self,
        identity: str,
        pose: str,
        samples: int,
    ) -> bool:
        if not self.capture_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "/face_recogniser/capture_face is unavailable"
            )
            return False

        goal = CaptureFace.Goal()
        goal.identity = identity
        goal.pose = pose
        goal.samples_required = samples

        self._last_feedback_state = ""

        goal_future = self.capture_client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback,
        )
        rclpy.spin_until_future_complete(
            self,
            goal_future,
        )

        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(
                f"{pose} capture goal was rejected"
            )
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self,
            result_future,
        )

        wrapped = result_future.result()
        if wrapped is None:
            self.get_logger().error(
                f"No result returned for {pose} capture"
            )
            return False

        result = wrapped.result

        if result.success:
            self.get_logger().info(
                result.message
            )
            return True

        self.get_logger().error(
            result.message
        )
        return False

    def commit(
        self,
        identity: str,
        relationship: str,
        preferred_address: str,
    ) -> bool:
        if not self.commit_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                "/face_recogniser/commit_enrolment is unavailable"
            )
            return False

        request = CommitFaceEnrollment.Request()
        request.identity = identity
        request.relationship = relationship
        request.preferred_address = preferred_address

        future = self.commit_client.call_async(
            request
        )
        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()
        if response is None:
            self.get_logger().error(
                "No response from commit service"
            )
            return False

        if response.success:
            self.get_logger().info(
                response.message
            )
            return True

        self.get_logger().error(
            response.message
        )
        return False


def prompt_for_pose(pose: str) -> None:
    instructions = {
        "front": "Look directly at K9.",
        "left": "Turn your head slightly to your left.",
        "right": "Turn your head slightly to your right.",
    }

    print()
    print(instructions[pose])
    input("Press Enter when ready... ")


def main(args=None) -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "identity",
        help="Name of person to enrol",
    )
    parser.add_argument(
        "--relationship",
        choices=("family", "friend"),
        default="friend",
    )
    parser.add_argument(
        "--preferred-address",
        default="",
        help=(
            "How K9 should address this person. "
            "Defaults to the identity for friends."
        ),
    )
    parser.add_argument(
        "--front-samples",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--left-samples",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--right-samples",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Start each pose immediately without waiting for Enter",
    )

    parsed, ros_args = parser.parse_known_args(args)

    preferred_address = parsed.preferred_address.strip()
    if not preferred_address:
        if parsed.relationship == "friend":
            preferred_address = parsed.identity.strip()
        else:
            parser.error(
                "--preferred-address is required for family enrolment"
            )

    rclpy.init(args=ros_args)
    node = FaceEnrollerClient()

    success = False

    try:
        captures = [
            ("front", parsed.front_samples),
            ("left", parsed.left_samples),
            ("right", parsed.right_samples),
        ]

        for pose, samples in captures:
            if not parsed.no_prompt:
                prompt_for_pose(pose)

            if not node.capture(
                parsed.identity,
                pose,
                samples,
            ):
                return

        success = node.commit(
            parsed.identity,
            parsed.relationship,
            preferred_address,
        )

    except KeyboardInterrupt:
        node.get_logger().warning(
            "Enrolment interrupted"
        )

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
