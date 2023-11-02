from ultralytics import YOLO
import cv2
from ai_42_vienna.vision.streamer import FrameStreamer

model = YOLO('ai_42_vienna/vision/models/yolov8n.pt')
    
def main():

    streamer = FrameStreamer(source=0, model=model)
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