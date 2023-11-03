import cv2
import torch
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

class FrameStreamer:
    def __init__(self, source, model):
        self.cap = cv2.VideoCapture(source)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.model = model

    def read(self):
        """
        Reads a frame from the video source.

        Args:
            None

        Returns:
            ret (bool): True if frame is read correctly, False otherwise.
            frame (ndarray): Frame read from the video source.
        """
        ret, frame = self.cap.read()
        return ret, frame

    def object_detection(self, frame):
        """
        Performs object detection on the frame.

        Args:
            frame (ndarray): Frame read from the video source.

        Returns:
            frame (ndarray): Frame with bounding boxes drawn around detected objects.
        """
        # Run object detection on the frame using the provided model
        results = self.model(frame)

        # Initialize people count to 0
        people_count = 0

        # Loop through each result from the object detection
        for result in results:
            # Get the bounding boxes and class names for each detected object
            boxes = result.boxes.cpu().numpy()
            class_names = result.boxes.cls

            # Loop through each bounding box and class name
            for i, box in enumerate(boxes):
                label = class_names[i]

                # If the detected object is not a person, skip to the next bounding box
                if label != 0:
                    continue

                # Increment the people count and draw a bounding box around the detected person
                people_count += 1
                r = box.xyxy[0].astype(int)
                cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)

        # Add the people count to the frame as text
        cv2.putText(frame, f"People Count: {people_count}", (10, 30), self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Return the frame with bounding boxes drawn around detected objects
        return frame
    

    def pose_estimation(self, frame):
        """
        Performs pose estimation on the frame.

        Args:
            frame (ndarray): Frame read from the video source.

        Returns:
            frame (ndarray): Frame with skeleton drawn around detected people.
        """

        # Initialize people count
        people_count = 0

        # Define skeleton
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        # Perform pose estimation on the frame
        results = self.model(frame)

        # Loop through the results
        for result in results:
            # Get the keypoints
            kpts = result.keypoints.cpu().numpy()
            kpts = kpts.xy[0].astype(int)
            nkpt, ndim = kpts.shape

            # If there are no keypoints, continue
            if nkpt == 0:
                continue
            else:
                # Increment people count
                people_count += 1

                # Loop through the keypoints
                for i, k in enumerate(kpts):
                    x_coord, y_coord = k[0], k[1]
                    if x_coord % frame.shape[1] != 0 and y_coord % frame.shape[0] != 0:
                        if len(k) == 3:
                            conf = k[2]
                            if conf < 0.5:
                                continue
                        cv2.circle(frame, (int(x_coord), int(y_coord)), 5, (255,255,255), -1, lineType=cv2.LINE_AA)

                    ndim = kpts.shape[-1]
                    # Loop through the skeleton
                    for i, sk in enumerate(skeleton):
                        pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                        pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                        if ndim == 3:
                            conf1 = kpts[(sk[0] - 1), 2]
                            conf2 = kpts[(sk[1] - 1), 2]
                            if conf1 < 0.5 or conf2 < 0.5:
                                continue
                        if pos1[0] % frame.shape[1] == 0 or pos1[1] % frame.shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                            continue
                        if pos2[0] % frame.shape[1] == 0 or pos2[1] % frame.shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                            continue
                        cv2.line(frame, pos1, pos2, (255,255,255), thickness=2, lineType=cv2.LINE_AA)

        # Add people count to the frame
        cv2.putText(frame, f"People Count: {people_count}", (10, 30), self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
    
    def depth_estimation(self, frame, transform):
        """
        Performs depth estimation on the frame.

        Args:
            frame (ndarray): Frame read from the video source.
            transform (torchvision.transforms): Transform to apply to the frame.

        Returns:
            depth_map (ndarray): Frame with depth map.
        """
        
        # Apply transformation to the frame
        input_data = transform(frame).to("mps")

        # Run model without gradient computation
        with torch.no_grad():
            # Get model prediction
            prediction = self.model(input_data)
            # Move prediction to CPU
            prediction = prediction.to("cpu")
            # Interpolate prediction to match frame shape
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert prediction tensor to numpy array
        output = prediction.cpu().numpy()

        # Normalize output values between 0 and 1
        depth_map = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Scale depth_map values to uint8 range (0-255)
        depth_map = (depth_map * 255).astype(np.uint8)

        # Apply color mapping to the depth_map
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        # Return final depth_map
        return depth_map


    def release(self):
        """
        Releases the video source.

        Args:
            None

        Returns:
            None
        """
        self.cap.release()

    @staticmethod
    def show_frame(frame):
        """
        Displays the frame.

        Args:
            frame (ndarray): Frame read from the video source.

        Returns:
            None
        """
        cv2.imshow('frame', frame)

    @staticmethod
    def destroy_windows():
        """
        Destroys all windows.

        Args:
            None

        Returns:
            None
        """
        cv2.destroyAllWindows()