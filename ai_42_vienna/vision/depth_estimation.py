import torch
import cv2
from ai_42_vienna.vision.streamer import FrameStreamer

def instantiate_model():

    model_types = ["DPT_Large", 
                   "DPT_Hybrid", 
                   "MiDaS_small"]

    model = model_types[2]

    midas = torch.hub.load("intel-isl/MiDaS", model)

    device = torch.device("mps")
    midas.to(device)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model == "DPT_Large" or model == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform

def main():

    midas, transform = instantiate_model()

    streamer = FrameStreamer(source=0, model=midas)
    while True:
        ret, frame = streamer.read()
        if not ret:
            break
        processed_frame = streamer.depth_estimation(frame, transform)
        streamer.show_frame(processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()