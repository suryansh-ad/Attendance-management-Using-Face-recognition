from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import shutil
from urllib.request import urlretrieve

import cv2
import numpy as np

from .config import (
    DATA_DIR,
    EMBEDDINGS_PATH,
    FACES_DIR,
    MODELS_DIR,
    SFACE_MODEL_PATH,
    YUNET_MODEL_PATH,
)


YUNET_MODEL_URL = "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
SFACE_MODEL_URL = "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"
MIN_DETECTION_SCORE = 0.8
MIN_FACE_SIZE = 120
MIN_SHARPNESS = 60.0
MIN_BRIGHTNESS = 55.0
DEFAULT_SIMILARITY_THRESHOLD = 0.34
DEFAULT_REQUIRED_CONSENSUS = 4
MAX_UNRECOGNIZED_FRAMES = 45
ENROLLMENT_DUPLICATE_THRESHOLD = 0.42


@dataclass
class RecognitionResult:
    student_id: int | None
    confidence: float
    recognized: bool = True
    message: str = ""


@dataclass
class EnrollmentResult:
    embeddings: list[np.ndarray]
    snapshots: list[np.ndarray]
    duplicate_student_id: int | None = None


class FaceEncodingStore:
    def __init__(self, faces_dir: Path = FACES_DIR, embeddings_path: Path = EMBEDDINGS_PATH) -> None:
        self.faces_dir = faces_dir
        self.embeddings_path = embeddings_path
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.embeddings_path.exists():
            return np.array([], dtype=np.int32), np.empty((0, 512), dtype=np.float32)
        payload = np.load(self.embeddings_path)
        return payload["student_ids"].astype(np.int32), payload["embeddings"].astype(np.float32)

    def save_student_encoding(self, student_id: int, enrollment: EnrollmentResult) -> None:
        student_dir = self.faces_dir / str(student_id)
        student_dir.mkdir(parents=True, exist_ok=True)

        for existing in student_dir.glob("*.png"):
            existing.unlink()

        for index, sample in enumerate(enrollment.snapshots, start=1):
            file_path = student_dir / f"sample_{index:02d}.png"
            cv2.imwrite(str(file_path), sample)

        student_ids, embeddings = self.load()
        if student_ids.size:
            keep_mask = student_ids != student_id
            student_ids = student_ids[keep_mask]
            embeddings = embeddings[keep_mask]

        new_ids = np.full(len(enrollment.embeddings), student_id, dtype=np.int32)
        new_embeddings = np.asarray(enrollment.embeddings, dtype=np.float32)

        if embeddings.size:
            student_ids = np.concatenate([student_ids, new_ids])
            embeddings = np.vstack([embeddings, new_embeddings])
        else:
            student_ids = new_ids
            embeddings = new_embeddings

        np.savez(self.embeddings_path, student_ids=student_ids, embeddings=embeddings)

    def delete_student_encoding(self, student_id: int) -> None:
        student_dir = self.faces_dir / str(student_id)
        if student_dir.exists():
            shutil.rmtree(student_dir, ignore_errors=True)

        student_ids, embeddings = self.load()
        if not student_ids.size:
            return

        keep_mask = student_ids != student_id
        remaining_ids = student_ids[keep_mask]
        remaining_embeddings = embeddings[keep_mask]

        if remaining_ids.size:
            np.savez(self.embeddings_path, student_ids=remaining_ids, embeddings=remaining_embeddings)
        elif self.embeddings_path.exists():
            self.embeddings_path.unlink()

    def find_matching_student(self, enrollment: EnrollmentResult, threshold: float = ENROLLMENT_DUPLICATE_THRESHOLD) -> int | None:
        student_ids, embeddings = self.load()
        if not student_ids.size or not enrollment.embeddings:
            return None

        sample_scores: list[tuple[int, float]] = []
        for current_embedding in enrollment.embeddings:
            similarities = embeddings @ current_embedding
            best_index = int(np.argmax(similarities))
            best_similarity = float(similarities[best_index])
            if best_similarity >= threshold:
                sample_scores.append((int(student_ids[best_index]), best_similarity))

        if not sample_scores:
            return None

        counts = Counter(student_id for student_id, _ in sample_scores)
        matched_student_id, matched_count = counts.most_common(1)[0]
        if matched_count < max(3, len(enrollment.embeddings) // 3):
            return None
        return matched_student_id


def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


@lru_cache(maxsize=1)
def _ensure_models() -> tuple[str, str]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not YUNET_MODEL_PATH.exists():
        try:
            urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_PATH)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the YuNet face detector model. Keep internet on and try again."
            ) from exc
    if not SFACE_MODEL_PATH.exists():
        try:
            urlretrieve(SFACE_MODEL_URL, SFACE_MODEL_PATH)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the SFace recognition model. Keep internet on and try again."
            ) from exc
    return str(YUNET_MODEL_PATH), str(SFACE_MODEL_PATH)


@lru_cache(maxsize=1)
def _detector():
    yunet_path, _ = _ensure_models()
    try:
        return cv2.FaceDetectorYN.create(
            yunet_path,
            "",
            (640, 640),
            score_threshold=0.7,
            nms_threshold=0.3,
            top_k=5000,
        )
    except AttributeError as exc:
        raise RuntimeError(
            "Your OpenCV build does not include FaceDetectorYN. Reinstall with requirements.txt."
        ) from exc


@lru_cache(maxsize=1)
def _recognizer():
    _, sface_path = _ensure_models()
    try:
        return cv2.FaceRecognizerSF.create(sface_path, "")
    except AttributeError as exc:
        raise RuntimeError(
            "Your OpenCV build does not include FaceRecognizerSF. Reinstall with requirements.txt."
        ) from exc


def _normalize(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _detect_faces(frame: np.ndarray) -> np.ndarray:
    detector = _detector()
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    return faces


def _primary_face(frame: np.ndarray):
    faces = _detect_faces(frame)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda face: float(face[2] * face[3]))


def _face_dimensions(face) -> tuple[int, int]:
    return int(face[2]), int(face[3])


def _extract_face_crop(frame: np.ndarray, face) -> np.ndarray | None:
    if face is None:
        return None
    aligned = _recognizer().alignCrop(frame, face)
    return aligned if aligned.size else None


def _embedding_from_face(frame: np.ndarray, face) -> np.ndarray | None:
    aligned = _extract_face_crop(frame, face)
    if aligned is None:
        return None
    feature = _recognizer().feature(aligned)
    if feature is None:
        return None
    return _normalize(feature.flatten())


def _quality_message(frame: np.ndarray, face) -> str | None:
    if face is None:
        return "Keep one clear face in frame"
    width, height = _face_dimensions(face)
    if min(width, height) < MIN_FACE_SIZE:
        return "Move closer to the camera"
    if float(face[-1]) < MIN_DETECTION_SCORE:
        return "Face not clear enough"
    if _sharpness(frame) < MIN_SHARPNESS:
        return "Image too blurry - hold still"
    if _brightness(frame) < MIN_BRIGHTNESS:
        return "Lighting is too low"
    return None


def enroll_from_camera(sample_count: int = 12) -> EnrollmentResult:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    store = FaceEncodingStore()
    collected_embeddings: list[np.ndarray] = []
    collected_snapshots: list[np.ndarray] = []
    duplicate_history: list[int] = []
    window_name = "Enrollment - Press C to capture, Q to quit"

    try:
        while len(collected_embeddings) < sample_count:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read from webcam.")

            face = _primary_face(frame)
            quality_issue = _quality_message(frame, face)
            message = quality_issue or f"Face ready - press C ({len(collected_embeddings)}/{sample_count})"

            if face is not None:
                x, y, w, h = face[:4].astype(int)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 0), 2)

            cv2.putText(frame, message, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 255), 2)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                raise RuntimeError("Enrollment cancelled.")

            if key == ord("c") and quality_issue is None and face is not None:
                face_crop = _extract_face_crop(frame, face)
                embedding = _embedding_from_face(frame, face)
                if face_crop is None or embedding is None:
                    continue

                probe_result = EnrollmentResult(embeddings=[embedding], snapshots=[])
                matched_student_id = store.find_matching_student(probe_result)
                if matched_student_id is not None:
                    duplicate_history.append(matched_student_id)
                    counts = Counter(duplicate_history[-3:])
                    duplicate_id, duplicate_count = counts.most_common(1)[0]
                    if duplicate_count >= 2:
                        return EnrollmentResult(
                            embeddings=[],
                            snapshots=[],
                            duplicate_student_id=duplicate_id,
                        )

                collected_snapshots.append(face_crop)
                collected_embeddings.append(embedding)

        return EnrollmentResult(embeddings=collected_embeddings, snapshots=collected_snapshots)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def recognize_from_camera(
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    required_consensus: int = DEFAULT_REQUIRED_CONSENSUS,
) -> RecognitionResult | None:
    store = FaceEncodingStore()
    student_ids, embeddings = store.load()
    if len(student_ids) == 0:
        raise RuntimeError("No enrolled faces found. Add students first.")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    history: list[int] = []
    scores_seen: list[float] = []
    consecutive_unknown = 0
    window_name = "Attendance Scan - Press Q to quit"

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read from webcam.")

            message = "Searching for enrolled face"
            face = _primary_face(frame)
            quality_issue = _quality_message(frame, face)

            if face is not None:
                x, y, w, h = face[:4].astype(int)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 0), 2)

            if quality_issue is None and face is not None:
                current_embedding = _embedding_from_face(frame, face)
                if current_embedding is None:
                    history.clear()
                    scores_seen.clear()
                    message = "Could not extract face features"
                    cv2.putText(frame, message, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 255), 2)
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        return None
                    continue
                similarities = embeddings @ current_embedding
                best_index = int(np.argmax(similarities))
                best_similarity = float(similarities[best_index])
                matched_student = int(student_ids[best_index])
                confidence = max(0.0, min(100.0, best_similarity * 100.0))

                if best_similarity >= similarity_threshold:
                    history.append(matched_student)
                    scores_seen.append(confidence)
                    consecutive_unknown = 0
                    counts = Counter(history[-required_consensus:])
                    student_id, count = counts.most_common(1)[0]
                    message = f"Recognizing face... similarity={best_similarity:.2f}"
                    if count >= required_consensus:
                        final_confidence = float(np.mean(scores_seen[-required_consensus:]))
                        return RecognitionResult(
                            student_id=student_id,
                            confidence=final_confidence,
                            recognized=True,
                            message="Student recognized.",
                        )
                else:
                    history.clear()
                    scores_seen.clear()
                    consecutive_unknown += 1
                    message = f"Face not recognized similarity={best_similarity:.2f}"
                    if consecutive_unknown >= MAX_UNRECOGNIZED_FRAMES:
                        return RecognitionResult(
                            student_id=None,
                            confidence=0.0,
                            recognized=False,
                            message="Student is not recognized.",
                        )
            else:
                history.clear()
                scores_seen.clear()
                if quality_issue is not None:
                    message = quality_issue

            cv2.putText(frame, message, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 255), 2)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return RecognitionResult(
                    student_id=None,
                    confidence=0.0,
                    recognized=False,
                    message="Attendance scan cancelled.",
                )
    finally:
        camera.release()
        cv2.destroyAllWindows()
