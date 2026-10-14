"""YuNet face localization + EmotiEffNet-B0 AFEW checkpoint, entirely local."""
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from realtime_fer.download_models import DEFAULT_MODELS, digest

LABELS_EN = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
LABELS_ZH = ['愤怒', '轻蔑', '厌恶', '恐惧', '高兴', '中性', '悲伤', '惊讶']


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (8,) or not np.isfinite(logits).all():
        raise ValueError(f'Unexpected emotion output: {logits.shape}')
    values = np.exp(logits - logits.max())
    return values / values.sum()


def preprocess(bgr):
    if bgr is None or bgr.size == 0:
        raise ValueError('Empty face crop')
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    # Same RGB ImageNet normalization and resize as the authors' ONNX implementation.
    x = rgb.astype(np.float32) / 255.
    x = (x - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, right - left) * max(0, bottom - top)
    areas = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(areas - inter, 1e-6)


@dataclass
class Track:
    box: tuple
    probabilities: np.ndarray
    last_seen: float


class Smoother:
    """Short-lived geometric tracks; IDs are not identities; no biometric storage."""
    def __init__(self, tau=.22, max_age=.6):
        self.tau, self.max_age = tau, max_age
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks.clear()
        self.next_id = 1

    def update(self, observations, timestamp):
        self.tracks = {key: value for key, value in self.tracks.items() if 0 <= timestamp - value.last_seen <= self.max_age}
        # Global greedy one-to-one matching prevents two faces sharing one history.
        pairs = sorted([(iou(box, tr.box), i, key) for i, (box, _) in enumerate(observations)
                        for key, tr in self.tracks.items()], reverse=True)
        matches, used = {}, set()
        for overlap, i, key in pairs:
            if overlap >= .3 and i not in matches and key not in used:
                matches[i] = key
                used.add(key)
        result = []
        for i, (box, probabilities) in enumerate(observations):
            key = matches.get(i)
            if key is None:
                key = self.next_id
                self.next_id += 1
            else:
                previous = self.tracks[key]
                dt = max(.001, timestamp - previous.last_seen)
                alpha = 1 - math.exp(-dt / self.tau)
                probabilities = (1 - alpha) * previous.probabilities + alpha * probabilities
            self.tracks[key] = Track(box, probabilities, timestamp)
            result.append(dict(track_id=key, box=box, probabilities=probabilities.copy()))
        return result


class Recognizer:
    def __init__(self, models=DEFAULT_MODELS, threads=4):
        import json
        models = Path(models)
        manifest_path = models / 'models.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'模型未安装：{models}\n请运行 realtime_fer.download_models。')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for filename in ['enet_b0_8_best_afew.onnx', 'face_detection_yunet_2023mar.onnx']:
            path = models / filename
            if filename not in manifest or not path.exists() or digest(path) != manifest[filename]['sha256']:
                raise ValueError(f'模型缺失或校验失败：{filename}')
        cv2.setNumThreads(max(1, threads))
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(threads, os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(models / 'enet_b0_8_best_afew.onnx'), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name
        self.detector = cv2.FaceDetectorYN.create(str(models / 'face_detection_yunet_2023mar.onnx'), '',
                                                (320, 320), .8, .3, 5000)
        self.smoother = Smoother()
        self.classify(np.zeros((64, 64, 3), dtype=np.uint8))  # verify output and warm up

    def classify(self, face):
        logits = self.session.run(None, {self.input: preprocess(face)})[0]
        return softmax(np.asarray(logits).reshape(-1))

    def reset(self):
        self.smoother.reset()

    def process(self, frame, timestamp=None, max_faces=4, smooth=True):
        height, width = frame.shape[:2]
        scale = min(1., 640 / max(height, width))
        small = cv2.resize(frame, (round(width * scale), round(height * scale))) if scale < 1 else frame
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        observations = []
        if faces is not None:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:max_faces]
            for face in faces:
                x, y, w, h = face[:4] / scale
                # Tight crop with a small context margin; no synthetic detail generation.
                x1, y1 = max(0, int(x - .03 * w)), max(0, int(y - .03 * h))
                x2, y2 = min(width, int(x + 1.03 * w)), min(height, int(y + 1.03 * h))
                if min(x2 - x1, y2 - y1) < 24:
                    continue
                box = (x1, y1, x2, y2)
                observations.append((box, self.classify(frame[y1:y2, x1:x2])))
        timestamp = time.monotonic() if timestamp is None else timestamp
        if smooth:
            return self.smoother.update(observations, timestamp)
        return [dict(track_id=i + 1, box=box, probabilities=prob) for i, (box, prob) in enumerate(observations)]
