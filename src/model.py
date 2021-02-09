import cv2
from src.mtcnn import MTCNN
import time


class FastMTCNN(object):
    """Fast MTCNN implementation."""

    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.

        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.

        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self, frames, save_faces, id, outdir, return_all):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                for f in frames
            ]
        if save_faces:
            save_paths = [f'{outdir}/{id}_{round(time.time() + i)}.jpg' for i in range(len(frames))]
        else:
            save_paths = None
        faces, probs, boxes, points = self.mtcnn(frames, save_path=save_paths, return_all=return_all)

        return faces, probs, boxes, points
