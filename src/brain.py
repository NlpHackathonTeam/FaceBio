import time

import cv2
import imutils
import numpy as np
from PIL import Image
from imutils.video import VideoStream
from tqdm import tqdm

from src.model import FastMTCNN
from utils import utils
from utils.profiler import FPS


class FaceDetection:
    def __init__(self, device, stride=4, resize=1, return_all=True, uuid=None, *args, **kwargs):
        self.device = device
        self.stride = stride
        self.resize = resize
        self.return_all = return_all
        self.fast_mtcnn = self._init_model(*args, **kwargs)
        self.frame_height, self.frame_width = None, None
        self.uuid = uuid

    def _init_model(self, *args, **kwargs):
        return FastMTCNN(
            stride=self.stride,
            resize=self.resize,
            device=self.device,
            *args, **kwargs
        )

    def run_detection_from_image(self, filenames, save_faces=True, outdir='./',
                                 plot_landmarks=False, show_frame=False):

        for filename in tqdm(filenames):

            # read the image
            image = Image.open(filename).convert('RGB')
            # create an image array copy so that we can use OpenCV functions on it
            image_array = np.array(image, dtype=np.float32)
            # cv2 image color conversion
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            faces, probs, bounding_boxes, landmarks = self.fast_mtcnn(image_array,
                                                                      save_faces=save_faces,
                                                                      id=self.uuid,
                                                                      outdir=outdir,
                                                                      return_all=self.return_all)

            # draw the bounding boxes around the faces
            try:
                image_array = utils.draw_bbox(bounding_boxes, image_array, probs[0])
                if plot_landmarks:
                    image_array = utils.plot_landmarks(landmarks, image_array)
            except:
                pass

            # show the image
            if show_frame:
                cv2.imshow('Image', image_array / 255.0)
                cv2.waitKey(0)

    def run_detection_from_video_file(self, video_file, outdir='./', save_faces=True,
                                      profiling=False, plot_landmarks=False):

        print("[INFO] Loading video file")
        cap = cv2.VideoCapture(video_file)
        profiler = FPS()
        if not cap.isOpened():
            print('Error while trying to read video. Please check path again')

        frame_count, total_fps, faces_detected = 0, 0, 0  # to count total frames

        # read until end of video
        while cap.isOpened():
            # capture each frame of the video
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                profiler.start()
                faces, probs, bounding_boxes, landmarks = self.fast_mtcnn(frame,
                                                                          save_faces=save_faces,
                                                                          id=self.uuid,
                                                                          outdir=outdir,
                                                                          return_all=self.return_all)
                if faces is not None:
                    faces_detected += len(faces)
                profiler.update(1)
                # color conversion for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # draw the bounding boxes around the faces
                try:
                    frame = utils.draw_bbox(bounding_boxes, frame, probs[0])
                    if plot_landmarks:
                        frame = utils.plot_landmarks(landmarks, frame)
                except:
                    pass

                cv2.imshow('Face detection frame', frame)
                # press `q` to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        profiler.stop()
        # release VideoCapture()
        print("[INFO] cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        # calculate and print the average FPS
        if profiling:
            print(f"Average FPS: {profiler.fps():.3f}")

    def run_detection_from_webcam_stream(self, save_faces=True, outdir='./', profiling=False,
                                         plot_landmarks=False):

        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        v_cap = VideoStream().start()
        profiler = FPS().start()
        time.sleep(2.0)

        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 600 pixels
            frame = v_cap.read()
            frame = imutils.resize(frame, width=600)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # grab the frame dimensions
            (self.frame_height, self.frame_width) = frame.shape[:2]
            faces, probs, bounding_boxes, landmarks = self.fast_mtcnn(frame, save_faces=save_faces,
                                                                      id=self.uuid,
                                                                      outdir=outdir,
                                                                      return_all=self.return_all)
            # color conversion for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            profiler.update(1)
            # draw the bounding boxes around the faces
            try:
                frame = utils.draw_bbox(bounding_boxes, frame, probs[0])
                if plot_landmarks:
                    frame = utils.plot_landmarks(landmarks, frame)
            except:
                pass
            # if the `q` key was pressed, break from the loop
            # show the output frame
            cv2.imshow("Output", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # do a bit of cleanup
        profiler.stop()
        print("[INFO] cleaning up...")
        cv2.destroyAllWindows()
        v_cap.stop()
        if profiling:
            print(f"Average FPS: {profiler.fps(), profiler._numFrames}")
