import datetime
import time


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self, numframes):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += numframes

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return round((self._numFrames / self.elapsed()) / 60, 4)


def fps(start_time, total_fps, frame_count, faces_detected):
    # get the end time
    end_time = time.time()
    # get the fps
    fps = 1 / (end_time - start_time)
    # add fps to total fps
    total_fps += fps
    # increment frame count
    frame_count += 1
    wait_time = max(1, int(fps / 4))
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    print(f"Total Faces Detected: {faces_detected}")