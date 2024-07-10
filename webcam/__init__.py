import base64
import traceback
from dataclasses import dataclass
from itertools import tee
from typing import Iterable, cast

import cv2
import mediapipe as mp
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python.components.containers import (Landmark,
                                                          NormalizedLandmark)
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

RunningMode = mp.tasks.vision.RunningMode
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
ImageFormat = mp.ImageFormat
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
Landmarks = list[NormalizedLandmark]

app = FastAPI(debug=True)

IMG_SIZE = np.array([640, 360])
DIST_COEFFS = np.zeros((4, 1))
CAM_MATRIX = np.array([
    [1, 0, .5],
    [0, 1, .5],
    [0, 0, 1]])


# POS_VEC = np.array([0, 0, 2])
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


@dataclass
class BBox:
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    @classmethod
    def from_mp(cls, mp_bbox):
        return cls(mp_bbox.origin_x, mp_bbox.origin_y, mp_bbox.width, mp_bbox.height)

    @property
    def xywh(self):
        return self.x, self.y, self.w, self.h

    @property
    def is_empty(self):
        return self.xywh == (0, 0, 0, 0)


@dataclass
class FLD:
    @classmethod
    def from_mp(cls, mp_fld: list[NormalizedLandmark]):
        return cls()


def lms2array(lms: list[Landmark]):
    return np.array([[lm.x, lm.y, lm.z] for lm in lms])


def norm2abs(frame: np.ndarray, point: np.ndarray):
    assert point.shape == (2,), point.shape
    return (point * frame.shape[:2]).astype(int)


def draw_head_pose(crop: np.ndarray | None, dir: np.ndarray | None, fld: FaceLandmarkerResult | None):
    try:
        assert crop is not None
        assert dir is not None
        assert fld is not None

        start = fld.face_landmarks[0][5]
        start = np.array([start.x, start.y])

        end = start + dir[:2] * 3

        start = norm2abs(crop, start)
        end = norm2abs(crop, end)

        print(start, end)

        cv2.line(crop, tuple(start), tuple(end), (255, 0, 0), 2)
    except Exception:
        traceback.print_exc()
        pass

    return crop


def draw_bbox(frame: np.ndarray, bbox: BBox | None):
    if bbox is None or bbox.is_empty:
        return frame

    x, y, w, h = bbox.xywh
    cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 2)
    return frame


def draw_pose_on_image(img: np.ndarray, pose: PoseLandmarkerResult | None):
    if pose is None:
        return img

    try:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # type: ignore
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(  # type: ignore
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in pose.pose_landmarks[0]])

        solutions.drawing_utils.draw_landmarks(  # type: ignore
            img,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,  # type: ignore
            solutions.drawing_styles.get_default_pose_landmarks_style())  # type: ignore

    except Exception:
        pass

    return img


def draw_landmarks_on_image(img: np.ndarray | None, fld: FaceLandmarkerResult | None):
    try:
        assert fld is not None

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # type: ignore
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(  # type: ignore
                x=landmark.x,
                y=landmark.y,
                z=landmark.z)
            for landmark in fld.face_landmarks[0]])

        solutions.drawing_utils.draw_landmarks(  # type: ignore
            image=img,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,  # type: ignore
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles  # type: ignore
            .get_default_face_mesh_tesselation_style())

        solutions.drawing_utils.draw_landmarks(  # type: ignore
            image=img,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,  # type: ignore
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles  # type: ignore
            .get_default_face_mesh_contours_style())

        solutions.drawing_utils.draw_landmarks(  # type: ignore
            image=img,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,  # type: ignore
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles  # type: ignore
            .get_default_face_mesh_iris_connections_style())
    except Exception:
        pass

    return img


def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        # print(frame.shape, frame.dtype)
        yield frame


def gen_small_frames(frames: Iterable[np.ndarray]):
    for frame in frames:
        yield cv2.resize(frame, tuple(IMG_SIZE))


def gen_pose(frames: Iterable[np.ndarray]):
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        output_segmentation_masks=False)

    with PoseLandmarker.create_from_options(options) as pose_landmarker:
        for frame in frames:
            try:
                img = mp.Image(
                    image_format=ImageFormat.SRGB,
                    data=frame)

                yield cast(
                    PoseLandmarkerResult,
                    pose_landmarker.detect(img))

            except Exception:
                traceback.print_exc()
                yield None


def gen_crop_from_pose(frames: Iterable[np.ndarray], poses: Iterable[PoseLandmarkerResult | None]):
    for frame, pose in zip(frames, poses):
        try:
            assert pose is not None

            lms = pose.pose_landmarks[0]
            lms = np.array([[lm.x, lm.y] for lm in lms[:10]])
            h, w, _ = frame.shape
            lms = (lms * [w, h]).astype(int)
            x1, y1 = lms.min(axis=0)
            x2, y2 = lms.max(axis=0)
            h, w = y2 - y1, x2 - x1

            yield frame[y1-h:y2+h, x1-w//2:x2+w//2, :].astype(np.uint8)

        except Exception:
            traceback.print_exc()
            yield None


def gen_bboxs(frames: Iterable[tuple[np.ndarray, PoseLandmarkerResult]]):
    face_detection_options = FaceDetectorOptions(
        base_options=python.BaseOptions('detector.tflite'),
        min_detection_confidence=0.5)

    with FaceDetector.create_from_options(face_detection_options) as face_detector:
        for frame, pose in frames:
            try:
                lms = lms2array(pose.pose_landmarks[0])
                img = mp.Image(
                    image_format=ImageFormat.SRGB,
                    data=frame)

                faces = face_detector.detect(img).detections

                face = faces[0]
                yield BBox.from_mp(face.bounding_box)
            except IndexError:
                yield None


def gen_head_poses(poses: Iterable[PoseLandmarkerResult | None]):
    for pose in poses:
        try:
            assert pose is not None

            lms = pose.pose_landmarks[0]
            lms = np.array([[lm.x, lm.y, lm.z] for lm in lms])
            m = (lms[7] + lms[8]) / 2
            s = lms[0]
            yield cast(np.ndarray, s - m)
        except Exception as e:
            if type(e) is not IndexError:
                traceback.print_exc()
            yield None


def gen_flds(frames: Iterable[np.ndarray | None]):
    fld_options = FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            'face_landmarker_v2_with_blendshapes.task'),
        running_mode=RunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)

    with FaceLandmarker.create_from_options(fld_options) as face_landmarker:
        for frame in frames:
            try:
                assert frame is not None

                img = mp.Image(
                    image_format=ImageFormat.SRGB,
                    data=frame)

                yield cast(
                    FaceLandmarkerResult,
                    face_landmarker.detect(img))

            except Exception:
                traceback.print_exc()
                yield None


def print_landmarks(landmarks: list[Landmark]):
    for i, landmark in enumerate(landmarks):
        print(f'Landmark {i}:')
        print(f' x: {landmark.x}')
        print(f' y: {landmark.y}')
        print(f' z: {landmark.z}')


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()

    frames = gen_frames()
    frames_crop, frames_small = tee(frames, 2)

    small_frames = gen_small_frames(frames_small)
    small_frames, small_frames_pose = tee(small_frames, 2)

    poses = gen_pose(small_frames_pose)
    poses, poses_crop, poses_pose = tee(poses, 3)

    head_poses = gen_head_poses(poses_pose)

    crops = gen_crop_from_pose(frames_crop, poses_crop)
    crops, crops_fld = tee(crops, 2)

    flds = gen_flds(crops_fld)

    for (
        small_frame,
        pose,
        head_pose,
        crop,
        fld,
    ) in zip(
        small_frames,
        poses,
        head_poses,
        crops,
        flds,
    ):

        small_frame = draw_pose_on_image(small_frame, pose)
        crop = draw_landmarks_on_image(crop, fld)
        crop = draw_head_pose(crop, head_pose, fld)

        def encode_img(img: np.ndarray | None):
            if img is None:
                return ''

            _, buffer = cv2.imencode('.jpg', img)
            return base64.b64encode(buffer).decode()

        await websocket.send_json({
            'img': encode_img(small_frame),
            'crop': encode_img(crop)})

if __name__ == '__main__':
    uvicorn.run(app,  host="127.0.0.1", port=8000)
