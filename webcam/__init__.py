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

IMG_SIZE = np.array([1440, 810])
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


def norm2abs(point: np.ndarray):
    assert point.shape == (2,), point.shape
    return (point * IMG_SIZE).astype(int)


def draw_line(frame: np.ndarray, start: np.ndarray | None, end: np.ndarray | None):
    try:
        assert start is not None and end is not None
        assert start.shape == end.shape == (2,)

        start = norm2abs(start)
        end = norm2abs(end)

        cv2.line(frame, tuple(start), tuple(end), (255, 0, 0), 2)
    except Exception:
        pass

    return frame


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


def draw_landmarks_on_image(img: np.ndarray, fld: FaceLandmarkerResult | None):
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
        yield cv2.resize(frame, tuple(IMG_SIZE))


def gen_bboxs(frames: Iterable[np.ndarray]):
    face_detection_options = FaceDetectorOptions(
        base_options=python.BaseOptions('detector.tflite'),
        min_detection_confidence=0.5)

    with FaceDetector.create_from_options(face_detection_options) as face_detector:
        for frame in frames:
            try:
                img = mp.Image(
                    image_format=ImageFormat.SRGB,
                    data=frame)

                faces = face_detector.detect(img).detections

                face = faces[0]
                yield BBox.from_mp(face.bounding_box)
            except IndexError:
                yield None


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
                yield None


def gen_head_poses(poses: Iterable[PoseLandmarkerResult | None]):
    for pose in poses:
        try:
            assert pose is not None

            lms = pose.pose_landmarks[0]
            lms = np.array([[lm.x, lm.y, lm.z] for lm in lms])
            m = (lms[7] + lms[8]) / 2
            s = lms[0]
            e = s + (s - m) * 2
            yield s[:2], e[:2]
        except Exception as e:
            if type(e) is not IndexError:
                print(traceback.format_exc())
            yield None, None


def gen_flds(frames: Iterable[np.ndarray]):
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
                img = mp.Image(
                    image_format=ImageFormat.SRGB,
                    data=frame)

                yield cast(
                    FaceLandmarkerResult,
                    face_landmarker.detect(img))

            except IndexError:
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
    frames, frames_bbox, frames_fld, frames_pos = tee(frames, 4)

    bboxs = gen_bboxs(frames_bbox)
    flds = gen_flds(frames_fld)

    poses = gen_pose(frames_pos)
    poses, poses_head = tee(poses, 2)

    head_poses = gen_head_poses(poses_head)

    for frame, bbox, fld, pose, (s, e) in zip(frames, bboxs, flds, poses, head_poses):
        frame = draw_bbox(frame, bbox)
        frame = draw_landmarks_on_image(frame, fld)
        frame = draw_pose_on_image(frame, pose)
        frame = draw_line(frame, s, e)

        bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        b64 = base64.b64encode(bytes).decode('utf-8')
        await websocket.send_json({
            'img': b64,
        })

if __name__ == '__main__':
    uvicorn.run(app,  host="127.0.0.1", port=8000)
