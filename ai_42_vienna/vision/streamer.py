import cv2

class FrameStreamer:
    def __init__(self, source, model):
        self.cap = cv2.VideoCapture(source)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.model = model

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def object_detection(self, frame):
        results = self.model(frame)
        people_count = 0
        for result in results:                                         
            boxes = result.boxes.cpu().numpy()
            class_names = result.boxes.cls

            for i, box in enumerate(boxes):                             
                label = class_names[i]
                if label != 0:                               
                    continue
                people_count += 1
                r = box.xyxy[0].astype(int)                           
                cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)  

        cv2.putText(frame, f"People Count: {people_count}", (10, 30), self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
    
    def pose_estimation(self, frame):

        people_count = 0

        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                         [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        
        results = self.model(frame)

        for result in results:
            kpts = result.keypoints.cpu().numpy()
            kpts = kpts.xy[0].astype(int)
            nkpt, ndim = kpts.shape

            if nkpt == 0:
                continue
            else:
                people_count += 1

                for i, k in enumerate(kpts):
                    x_coord, y_coord = k[0], k[1]
                    if x_coord % frame.shape[1] != 0 and y_coord % frame.shape[0] != 0:
                        if len(k) == 3:
                            conf = k[2]
                            if conf < 0.5:
                                continue
                        cv2.circle(frame, (int(x_coord), int(y_coord)), 5, (255,255,255), -1, lineType=cv2.LINE_AA)
                    
                    ndim = kpts.shape[-1]
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


        cv2.putText(frame, f"People Count: {people_count}", (10, 30), self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def release(self):
        self.cap.release()

    @staticmethod
    def show_frame(frame):
        cv2.imshow('frame', frame)

    @staticmethod
    def destroy_windows():
        cv2.destroyAllWindows()