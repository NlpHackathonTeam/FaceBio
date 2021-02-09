# FaceBIO - Face Recognition Using Mtcnn
This is a repository for face detection on edge devices using an efficient pre-trained pytorch implementation of MTCNN for face detection.

## Table of contents

* [Table of contents](#table-of-contents)
* [Quick start](#quick-start)
* [References](#references)

## Quick start

1. Install:

    ```bash
    # clone this repo, removing the '-' to allow python imports:
    git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch
    
    # create a virtual environment
    python3 -m venv path_to_virtual_envs/name_of_venv
    
    # activate path
    source path_to_virtual_envs/name_of_venv/bin/activate
    
    # install the required packages
    pip install -r requirements
    
    ```

2. How to run:

    ```bash
    python Inference.py --mode video --input video_file --outDir folder_to_save_screenshots --config configs/config.yaml --used-id user1
    ```
   For now this repo supports the following functionalities:

        * Face detection in simple image file: use --mode image
        * Face detection in simple video file: use --mode video
        * Real time Face detection by using webcam: use --mode webcam no --input is required
        * Reat time Face detection by Rasberry Pi cam: Not implemented yet  

   * In order to be able to run the recipe you need to define a __yaml configuration__ file.  
   A configuration with the baseline parameteres is included in this repo but feel free to create your own.

   * Also, is not required but it is preferable to pass a __user-id__ every time you call it in order to make proper
   output names for the detected faces: userID_timestamp.jpg

3. Further Steps needed:

   * Dockerize the repository
   * Add [jetcam](https://github.com/NVIDIA-AI-IOT/jetcam) for more configurable use of csi or usb camera.
   * Add a proper logger
   * Run proper tests


## References

1. David Sandberg's facenet repo: [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)

2. F. Schroff, D. Kalenichenko, J. Philbin. _FaceNet: A Unified Embedding for Face Recognition and Clustering_, arXiv:1503.03832, 2015. [PDF](https://arxiv.org/pdf/1503.03832)

3. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. _VGGFace2: A dataset for recognising face across pose and age_, International Conference on Automatic Face and Gesture Recognition, 2018. [PDF](http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf)

4. D. Yi, Z. Lei, S. Liao and S. Z. Li. _CASIAWebface: Learning Face Representation from Scratch_, arXiv:1411.7923, 2014. [PDF](https://arxiv.org/pdf/1411.7923)

5. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. _Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks_, IEEE Signal Processing Letters, 2016. [PDF](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)
6. Tim Esler's facenet-pytorch repo: [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)

