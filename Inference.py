import glob
import time
import cv2
import torch
from imutils.video import FileVideoStream
from tqdm import tqdm
from model import FastMTCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

filenames = glob.glob('/Users/igkinis/Desktop/projects/faceBIO/data/videos/*.mp4')[1:2]

frames = []
frames_processed = 0
faces_detected = 0
batch_size = 60
start = time.time()

fast_mtcnn = FastMTCNN(
    stride=4,
    resize=1,
    margin=20,
    factor=0.6,
    keep_all=True,
    post_process=False,
    device=device
)


def run_detection(fast_mtcnn, filenames, save_faces, outdir, profiling=False):
    frames = []
    frames_processed = 0
    faces_detected = 0
    batch_size = 60
    start = time.time()

    for filename in tqdm(filenames):

        v_cap = FileVideoStream(filename).start()
        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        for j in range(v_len):

            frame = v_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            #if len(frames) >= batch_size or j == v_len - 1:
        faces = fast_mtcnn(frames, save_faces=save_faces,
                           id=filename.split("/")[-1].split(".")[0], outdir=outdir)

        frames_processed += len(frames)
        faces_detected += len(faces)
        frames = []

        if profiling:
            print(
                f'Frames per second: {frames_processed / (time.time() - start):.3f},',
                f'faces detected: {faces_detected}\r'
            )

        v_cap.stop()


run_detection(fast_mtcnn, filenames=filenames, save_faces=True,
              outdir='/Users/igkinis/Desktop/projects/faceBIO/data', profiling=True)
