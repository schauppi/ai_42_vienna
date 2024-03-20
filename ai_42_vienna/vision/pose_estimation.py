from ultralytics import YOLO
import cv2
from ai_42_vienna.vision.streamer import FrameStreamer


def instantiate_model():
    """
    Instantiate the YOLO model for pose estimation

    Args:
        None

    Returns:
        yolo_pose: YOLO model for pose estimation
    """
    yolo_pose = YOLO('ai_42_vienna/vision/models/yolov8n-pose.pt')

    return yolo_pose


def main():
    """
    Main function for pose estimation

    Args:
        None

    Returns:
        None
    """

    yolo_pose = instantiate_model()

    streamer = FrameStreamer(source=0, object_detection_model=None,
                             pose_estimation_model=yolo_pose, depth_estimation_model=None)
    while True:
        ret, frame = streamer.read()
        if not ret:
            break
        processed_frame = streamer.pose_estimation(frame)
        streamer.show_frame(processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    streamer.release()
    streamer.destroy_windows()


if __name__ == '__main__':
    main()
