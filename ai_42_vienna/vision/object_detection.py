from ultralytics import YOLO
import cv2
from ai_42_vienna.vision.streamer import FrameStreamer


def instantiate_model():
    """
    Instantiate the YOLO model for object detection

    Args:
        None

    Returns:
        yolo: YOLO model for object detection
    """

    yolo = YOLO('ai_42_vienna/vision/models/yolov8n.pt')

    return yolo


def main():
    """
    Main function for object detection 

    Args:
        None

    Returns:    
        None
    """

    yolo = instantiate_model()

    streamer = FrameStreamer(source=0, object_detection_model=yolo,
                             pose_estimation_model=None, depth_estimation_model=None)
    while True:
        ret, frame = streamer.read()
        if not ret:
            break
        processed_frame = streamer.object_detection(frame)
        streamer.show_frame(processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    streamer.release()
    streamer.destroy_windows()


if __name__ == '__main__':
    main()
