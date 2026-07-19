"""Image forensics checks: ELA, copy-move, halftone, masthead matching. Pure OpenCV, no ML models."""

import cv2 as _cv2

# These checks run on modest single-image inputs, not video - OpenCV's own
# OpenCL (T-API) dispatch buys nothing here and has a known teardown crash
# on some Intel iGPU driver stacks (observed: clReleaseDevice failing at
# process exit once `openvino` also touches the OpenCL platform). Disabling
# it is a one-line, zero-behavior-change fix; real NPU/iGPU acceleration
# for this project goes through OpenVINO (ml/openvino_runtime.py), not
# OpenCV's OpenCL backend.
_cv2.ocl.setUseOpenCL(False)
