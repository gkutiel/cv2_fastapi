import base64
import traceback
from dataclasses import dataclass
from itertools import tee
from typing import Iterable, cast

import cv2
import mediapipe as mp
import numpy as np
import transforms3d.affines as affines
import uvicorn
from cattrs import unstructure
from fastapi import FastAPI, WebSocket
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python.components.containers import (Landmark,
                                                          NormalizedLandmark)
from mediapipe.tasks.python.vision.face_detector import FaceDetectorResult
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

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


def encode_img(img: np.ndarray | None):
    try:
        assert img is not None
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode()
    except Exception:
        return ''


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
    h, w, _ = frame.shape
    return (point * [w, h]).astype(int)


def draw_head_pose(crop: np.ndarray | None, dir: np.ndarray | None, fld: FaceLandmarkerResult | None):
    try:
        assert crop is not None
        assert dir is not None
        assert fld is not None

        start = fld.face_landmarks[0][4]
        start = np.array([start.x, start.y])

        end = start + dir[:2] * 3

        start = norm2abs(crop, start)
        end = norm2abs(crop, end)

        cv2.line(crop, tuple(start), tuple(end), (255, 0, 0), 2)
    except Exception:
        traceback.print_exc()

    return crop


def draw_bbox(frame: np.ndarray, bbox: BBox | None):
    if bbox is None or bbox.is_empty:
        return frame

    x, y, w, h = bbox.xywh
    cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 2)
    return frame


def draw_pose_on_image(img: np.ndarray | None, pose: PoseLandmarkerResult | None):
    try:
        assert pose is not None

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
        traceback.print_exc()

    return img


def draw_landmarks_2d(img: np.ndarray | None, lms: np.ndarray):
    try:
        assert img is not None
        assert lms.shape[1] == 2

        h, w, _ = img.shape
        for x, y in lms:
            x, y = int(x * w), int(y * h)
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

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
        try:
            _, frame = cap.read()
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            traceback.print_exc()
            yield None


def gen_small_frames(frames: Iterable[np.ndarray | None]):
    for frame in frames:
        try:
            assert frame is not None
            yield cv2.resize(frame, tuple(IMG_SIZE))
        except Exception:
            traceback.print_exc()
            yield None


def gen_faces(frames: Iterable[np.ndarray | None]):
    face_detection_options = FaceDetectorOptions(
        base_options=python.BaseOptions('detector.tflite'),
        min_detection_confidence=0.5)

    with FaceDetector.create_from_options(face_detection_options) as face_detector:
        for frame in frames:
            try:
                assert frame is not None

                img = mp.Image(
                    image_format=ImageFormat.SRGB,
                    data=frame)

                yield cast(
                    FaceDetectorResult,
                    face_detector.detect(img))
            except Exception:
                traceback.print_exc()
                yield None


def gen_pose(frames: Iterable[np.ndarray | None]):
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = PoseLandmarkerOptions(
        base_options=base_options,
        min_pose_detection_confidence=0.45,
        running_mode=RunningMode.IMAGE,
        output_segmentation_masks=False)

    with PoseLandmarker.create_from_options(options) as pose_landmarker:
        for frame in frames:
            try:
                assert frame is not None

                img = mp.Image(
                    image_format=ImageFormat.SRGB,
                    data=frame)

                yield cast(
                    PoseLandmarkerResult,
                    pose_landmarker.detect(img))

            except Exception:
                traceback.print_exc()
                yield None


def gen_crop_from_pose(frames: Iterable[np.ndarray | None], poses: Iterable[PoseLandmarkerResult | None]):
    for frame, pose in zip(frames, poses):
        try:
            assert pose is not None
            assert frame is not None

            lms = pose.pose_landmarks[0]
            lms = np.array([[lm.x, lm.y] for lm in lms[:10]])
            h, w, _ = frame.shape
            lms = (lms * [w, h]).astype(int)
            x1, y1 = lms.min(axis=0)
            x2, y2 = lms.max(axis=0)
            h, w = y2 - y1, x2 - x1

            crop = frame = frame[
                y1-h:y2+h,
                x1-w//2:x2+w//2, :].astype(np.uint8)

            h, w, _ = crop.shape
            d = max(h, w)
            pad = np.ones((d, d, 3), dtype=np.uint8) * 255
            pad[:h, :w] = crop

            yield pad

        except Exception:
            print('CROP ERROR')
            traceback.print_exc()
            yield None


def gen_trans(poses: Iterable[PoseLandmarkerResult | None]):
    for pose in poses:
        try:
            assert pose is not None

            # lms = lms2array(pose.pose_world_landmarks[0])
            lms = lms2array(pose.pose_landmarks[0])
            yield lms[0]
        except Exception:
            traceback.print_exc()
            yield np.zeros(3)


def gen_flds(crops: Iterable[np.ndarray | None]):
    fld_options = FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            'face_landmarker_v2_with_blendshapes.task'),
        running_mode=RunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1)

    with FaceLandmarker.create_from_options(fld_options) as face_landmarker:
        for frame in crops:
            try:
                assert frame is not None
                assert all(d > 0 for d in frame.shape)

                img = mp.Image(
                    image_format=ImageFormat.SRGB,
                    data=frame)

                yield cast(
                    FaceLandmarkerResult,
                    face_landmarker.detect(img))

            except Exception:
                traceback.print_exc()
                yield None


def gen_norm_flds_2d(flds: Iterable[FaceLandmarkerResult | None]):
    for fld in flds:
        try:
            assert fld is not None

            tmat = fld.facial_transformation_matrixes[0]

            T, R, *_ = affines.decompose(tmat)
            print('T', T)
            # rot = Rotation.from_matrix(rot).as_euler('xyz', degrees=False)
            R_inv = np.linalg.inv(R)

            # print('ROT', rot.shape)
            # yield rot

            lms = fld.face_landmarks[0]
            lms = lms2array(fld.face_landmarks[0])
            # lms = lms * 30
            print('LMS', lms[0])

            # tmat_inv = np.linalg.inv(tmat)
            # lms = np.hstack([lms, np.ones((lms.shape[0], 1))])
            # lms_norm = (R_inv @ lms.T).T - (R_inv @ T)
            lms_norm = (R @ lms.T).T
            # lms_norm = (R @ lms.T).T + T
            # lms_norm = lms_norm / 30

            print('LMS_NORM', lms_norm[0])
            yield lms_norm[:, :2]

        except Exception:
            traceback.print_exc()
            yield np.zeros(3)


def ears(fld: FaceLandmarkerResult | None):
    try:
        assert fld is not None

        # left right top bottom
        le = np.array([362, 263, 386, 374])
        re = np.array([33, 133, 159, 145])

        lms = lms2array(fld.face_landmarks[0])

        def ear(eye):
            left, right, top, bottom = eye
            h = norm(top - bottom)
            w = norm(left - right)
            return h / w

        return ear(lms[le]), ear(lms[re])

    except Exception:
        traceback.print_exc()
        return 0, 0


@dataclass(kw_only=True)
class Msg:
    frame_src: str
    face_src: str
    trans: list
    rot: list
    ear_left: str
    ear_right: str


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()

    frames = gen_frames()
    frames_crop, frames_small = tee(frames, 2)

    small_frames = gen_small_frames(frames_small)
    small_frames, small_frames_pose, small_frames_faces = tee(small_frames, 3)

    faces = gen_faces(small_frames_faces)

    poses = gen_pose(small_frames_pose)
    poses, poses_crop, poses_trans = tee(poses, 3)

    trans = gen_trans(poses_trans)

    crops = gen_crop_from_pose(frames_crop, poses_crop)
    crops, crops_fld = tee(crops, 2)

    flds = gen_flds(crops_fld)
    flds, flds_norm = tee(flds, 2)

    norm_flds = gen_norm_flds_2d(flds_norm)
    # rots = gen_rots(flds_rots)

    for (
        small_frame,
        face,
        pose,
        crop,
        fld,
        trans,
        norm_fld
    ) in zip(
            small_frames,
            faces,
            poses,
            crops,
            flds,
            trans,
            norm_flds):

        try:
            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR)
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

            small_frame = draw_pose_on_image(small_frame, pose)
            crop = draw_landmarks_on_image(crop, fld)
            crop = draw_landmarks_2d(crop, norm_fld)
            # crop = draw_head_pose(crop, head_pose, fld)

            # yaw, pitch = yaw_pitch(head_pose)

            ear_left, ear_right = ears(fld)

            msg = Msg(
                frame_src=encode_img(small_frame),
                face_src=encode_img(crop),
                ear_left=f'{ear_left:.2f}',
                ear_right=f'{ear_right:.2f}',
                trans=np.round(trans, 2).tolist(),
                rot=[]
                # rot=np.round(rots, 2).tolist()
            )

            msg = unstructure(msg)

            ear_left = await websocket.send_json(msg)

        except Exception:
            traceback.print_exc()
            msg = Msg(
                frame_src=encode_img(small_frame),
                face_src=encode_img(np.zeros((1, 1, 3), dtype=np.uint8)),
                ear_left='-',
                ear_right='-',
                trans=[],
                rot=[])

            msg = unstructure(msg)
            await websocket.send_json(msg)


if __name__ == '__main__':
    uvicorn.run(app,  host="127.0.0.1", port=8000)
