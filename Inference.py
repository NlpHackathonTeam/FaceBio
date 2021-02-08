import argparse
import glob

import torch

from brain import FaceDetection

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    face_detection = FaceDetection(
        stride=4,
        resize=1,
        margin=20,
        factor=0.6,
        keep_all=True,
        post_process=False,
        confidence=0.9,
        min_face_size=40,
        device=device,
        uuid=args.uuid
    )

    if args.mode == 'image':
        filenames = glob.glob(args.input)
        face_detection.run_detection_from_image(filenames=filenames, save_faces=True,
                                                outdir=args.outDir, plot_landmarks=True)
    elif args.mode == 'video':
        face_detection.run_detection_from_video_file(args.input, save_faces=True,
                                                     outDir=args.outDir, profiling=True)
    elif args.mode == 'webcam':
        face_detection.run_detection_from_webcam_stream(save_faces=True, outdir=args.outDir,
                                                        profiling=True, plot_landmarks=True)
    elif args.mode == 'Picam':
        print("Not implemented yet")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script splits a dataset to enrol and "
                                                 "verification based on a file with format"
                                                 "[counts of audios][spkID] and a standard wav.scp"
                                                 "file. It assumes that sox in installed on "
                                                 "your system")
    parser.add_argument("-i", "--input", type=str, required=False,
                        #default='/Users/igkinis/Desktop/projects/faceBIO/data/lfw/lfw/Aaron_Guiel/Aaron_Guiel_0001.jpg',
                        default='/Users/igkinis/Desktop/projects/faceBIO/data/input/video2.mp4',
                        help="Input speaker ids file (with counts of clips)")
    parser.add_argument("-m", "--mode", type=str, required=False,
                        default='webcam',
                        help="A flag to determine if the video is either a local file or a "
                             "streaming from a camera | image, video, webcam, Picam")
    parser.add_argument("-c", "--confidence", type=float, required=False,
                        default=0.5,
                        help="minimum probability to filter weak detections")
    parser.add_argument("-o", "--outDir", type=str, required=False,
                        default='/Users/igkinis/Desktop/projects/faceBIO/data',
                        help="A directory to save detected images files")
    parser.add_argument("-conf", "--config", type=str, required=False,
                        default='',
                        help="A configuration to parse model parameters")
    parser.add_argument("-uuid", "--user-id", type=str, required=False,
                        default='user1',
                        help="A unique user identifier to use for screenshots naming")
    args = parser.parse_args()
    main(args)
