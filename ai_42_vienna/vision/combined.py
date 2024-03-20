from ultralytics import YOLO
import cv2
from ai_42_vienna.vision.streamer import FrameStreamer
import copy


def instantiate_models():
    """
    Instantiate the YOLO model for pose estimation and the YOLO model for object detection

    Args:
        None

    Returns:
        yolo_pose: YOLO model for pose estimation
        yolo_object: YOLO model for object detection
    """
    yolo_pose = YOLO('ai_42_vienna/vision/models/yolov8n-pose.pt')
    yolo_object = YOLO('ai_42_vienna/vision/models/yolov8n.pt')

    return yolo_pose, yolo_object


def main():
    """
    Main function for combined vision

    Args:
        None

    Returns:
        None
    """

    yolo_pose, yolo_object = instantiate_models()

    streamer = FrameStreamer(source=0, object_detection_model=yolo_object,
                             pose_estimation_model=yolo_pose, depth_estimation_model=None)
    while True:
        ret, frame = streamer.read()
        frame_object = copy.deepcopy(frame)
        frame_pose = copy.deepcopy(frame)
        if not ret:
            break
        processed_frame_pose = streamer.object_detection(frame_object)
        processed_frame_object = streamer.pose_estimation(frame_pose)

        streamer.show_frames_side_by_side(
            processed_frame_object, processed_frame_pose)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    streamer.release()
    streamer.destroy_windows()
