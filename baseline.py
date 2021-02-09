import time
from imutils.video import VideoStream
from src.model import FastMTCNN
import imutils
from utils import utils
import cv2

device = 'cpu'
fast_mtcnn= FastMTCNN(
            stride=4,
            resize=1,
            margin=20,
            factor=0.6,
            min_face_size=40,
            keep_all=True,
            post_process=False,
            device=device
        )
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
v_cap = VideoStream().start()

# Define the codec and create VideoWriter object
time.sleep(2.0)

# define codec and create VideoWriter object
frames = []
frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = v_cap.read()
    frame = imutils.resize(frame, width=400)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # pil_image = Image.fromarray(frame).convert('RGB')
    pil_image = frame
    # grab the frame dimensions and convert it to a blob
    (frame_height, frame_width) = frame.shape[:2]
    out = cv2.VideoWriter('/Users/igkinis/Desktop/projects/faceBIO/data/output5.avi', fourcc, 20.0, (frame_width, frame_height))
    faces, probs, bounding_boxes = fast_mtcnn(pil_image, save_faces=True,
                            id="test_video", outdir='/Users/igkinis/Desktop/projects/faceBIO/data', return_prob=True)
    # color conversion for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # draw the bounding boxes around the faces
    try:
        frame = utils.draw_bbox(bounding_boxes, frame, probs[0])
    except:
        pass
    # if the `q` key was pressed, break from the loop
    # show the output frame
    cv2.imshow("Output", frame)
    output = frame
    out.write(output)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
v_cap.stop()
out.release()