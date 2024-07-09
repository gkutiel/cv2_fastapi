from ast import List
from dataclasses import dataclass
from itertools import tee
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult

RunningMode = mp.tasks.vision.RunningMode
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
ImageFormat = mp.ImageFormat


app = FastAPI(debug=True)

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


def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        h, w, *_ = frame.shape
        yield cv2.resize(frame, (w//2, h//2))


def gen_bboxs(frames: Iterable[np.ndarray]):
    face_detection_options = FaceDetectorOptions(
        base_options=python.BaseOptions('detector.tflite'),
        min_detection_confidence=0.5)

    with FaceDetector.create_from_options(face_detection_options) as face_detector:
        for frame in frames:
            img = mp.Image(
                image_format=ImageFormat.SRGB,
                data=frame)

            faces = face_detector.detect(img).detections

            if not faces:
                yield BBox()

            if faces:
                face = faces[0]
                yield BBox.from_mp(face.bounding_box)


def gen_flds(frames: Iterable[tuple[np.ndarray, BBox]]):
    fld_options = FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            'face_landmarker_v2_with_blendshapes.task'),
        running_mode=RunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)

    with FaceLandmarker.create_from_options(fld_options) as face_landmarker:
        for frame, bbox in frames:
            if bbox.is_empty:
                yield []
                continue

            img = mp.Image(
                image_format=ImageFormat.SRGB,
                data=frame)

            flds = face_landmarker.detect(img).face_landmarks

            if not flds:
                yield []
                continue

            yield flds[0]


def draw_bbox(frame: np.ndarray, bbox: BBox):
    x, y, w, h = bbox.xywh
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame


def draw_landmarks_on_image(img: np.ndarray, fld: list[NormalizedLandmark]):
    if not fld:
        return img

    annotated_image = np.copy(img)

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # type: ignore
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(  # type: ignore
            x=landmark.x,
            y=landmark.y,
            z=landmark.z)
        for landmark in fld])

    solutions.drawing_utils.draw_landmarks(  # type: ignore
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,  # type: ignore
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles  # type: ignore
        .get_default_face_mesh_tesselation_style())

    solutions.drawing_utils.draw_landmarks(  # type: ignore
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,  # type: ignore
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles  # type: ignore
        .get_default_face_mesh_contours_style())

    solutions.drawing_utils.draw_landmarks(  # type: ignore
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,  # type: ignore
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles  # type: ignore
        .get_default_face_mesh_iris_connections_style())

    return annotated_image


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    frames = gen_frames()
    frames, frames1, frames2 = tee(frames, 3)

    bboxs = gen_bboxs(frames1)
    bboxs, bboxs1 = tee(bboxs, 2)

    flds = gen_flds(zip(frames2, bboxs1))

    for frame, bbox, fld in zip(frames, bboxs, flds):
        frame = draw_bbox(frame, bbox)
        frame = draw_landmarks_on_image(frame, fld)

        await websocket.send_bytes(cv2.imencode('.jpg', frame)[1].tobytes())

if __name__ == '__main__':
    uvicorn.run(app,  host="127.0.0.1", port=8000)
