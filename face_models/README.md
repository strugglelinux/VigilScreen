YuNet face detector: `yunet.onnx` is OpenCV Zoo's `face_detection_yunet_2023mar.onnx`.
Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
Download: https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
License: see LICENSE (MIT).
The application uses a fixed 320×320 input with aspect-preserving padding.

SFace identity verification: `sface_int8.onnx` is OpenCV Zoo's `face_recognition_sface_2021dec_int8.onnx`.
Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
Download: https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx
License: see SFACE_LICENSE (Apache 2.0).
Enrollment images and computed identity features stay on this machine.

Face/iris landmarks: `face_landmarker.task` from Google's MediaPipe model bundle.
Source: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
Used locally to estimate eye opening and iris position; the application adds its own gaze geometry and calibration rules.
