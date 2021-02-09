import argparse
import glob

import torch

from src.brain import FaceDetection
from utils.config import Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    # create an instance of configuration file
    cfg = Config.from_yaml(args.config)

    face_detection = FaceDetection(
        stride=cfg.config["params"]["stride"],
        resize=cfg.config["params"]["resize"],
        margin=cfg.config["model"]["margin"],
        factor=cfg.config["model"]["factor"],
        keep_all=cfg.config["model"]["keep_all"],
        post_process=cfg.config["model"]["post_process"],
        confidence=cfg.config["model"]["confidence"],
        min_face_size=cfg.config["model"]["min_face_size"],
        return_all=cfg.config["params"]["return_all"],
        convert2rt=cfg.config["model"]["convert2rt"],
        load2rt=cfg.config["model"]["load2rt"],
        device=device,
        uuid=args.user_id
    )

    if args.mode == 'image':
        filenames = glob.glob(args.input)
        face_detection.run_detection_from_image(filenames=filenames,
                                                save_faces=cfg.config["params"]["save_faces"],
                                                outdir=args.outDir,
                                                plot_landmarks=cfg.config["params"][
                                                    "plot_landmarks"])
    elif args.mode == 'video':
        face_detection.run_detection_from_video_file(args.input,
                                                     save_faces=cfg.config["params"]["save_faces"],
                                                     outdir=args.outDir,
                                                     profiling=cfg.config["params"]["profiling"],
                                                     plot_landmarks=cfg.config["params"][
                                                         "plot_landmarks"])
    elif args.mode == 'webcam':
        face_detection.run_detection_from_webcam_stream(
            save_faces=cfg.config["params"]["save_faces"], outdir=args.outDir,
            profiling=cfg.config["params"]["profiling"],
            plot_landmarks=cfg.config["params"]["plot_landmarks"])
    elif args.mode == 'Picam':
        print("Not implemented yet")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script splits a dataset to enrol and "
                                                 "verification based on a file with format"
                                                 "[counts of audios][spkID] and a standard wav.scp"
                                                 "file. It assumes that sox in installed on "
                                                 "your system")
    parser.add_argument("-i", "--input", type=str, required=False,
                        default='/Users/igkinis/Desktop/projects/faceBIO/data/lfw/lfw/Aaron_Guiel/Aaron_Guiel_0001.jpg',
                        #default='/Users/igkinis/Desktop/projects/faceBIO/data/input/video2.mp4',
                        help="Input speaker ids file (with counts of clips)")
    parser.add_argument("-m", "--mode", type=str, required=True,
                        default='image',
                        help="A flag to determine if the video is either a local file or a "
                             "streaming from a camera | image, video, webcam, Picam")
    parser.add_argument("-o", "--outDir", type=str, required=False,
                        default='/Users/igkinis/Desktop/projects/faceBIO/data/igkinis',
                        help="A directory to save detected images files")
    parser.add_argument("-conf", "--config", type=str, required=True,
                        default='/Users/igkinis/Desktop/projects/faceBIO/configs/config.yaml',
                        help="A configuration to parse model parameters")
    parser.add_argument("-uuid", "--user-id", type=str, required=False,
                        default='user1',
                        help="A unique user identifier to use for screenshots naming")
    args = parser.parse_args()
    main(args)
