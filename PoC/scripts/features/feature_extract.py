"""Halpe-26 좌표에서 연구 근거가 있는 러닝 동작 지표를 계산한다.

고정된 측면 영상을 전제로 하며, 착지와 발끝 이탈 시점은 2D 좌표 변화로
추정한다. 따라서 시간 관련 결과는 포스 플레이트 측정값이 아닌 추정값이다.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


KEYPOINTS = {
    "nose": 0, "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8, "left_wrist": 9,
    "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15,
    "right_ankle": 16, "neck": 18, "hip_center": 19,
    "left_big_toe": 20, "right_big_toe": 21,
    "left_small_toe": 22, "right_small_toe": 23,
    "left_heel": 24, "right_heel": 25,
}


@dataclass(frozen=True)
class Contact:
    frame: int
    side: str
    prominence: float


def _finite_point(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all())


def _angle(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    """vertex를 꼭짓점으로 하는 0~180도 사이의 작은 각을 반환한다."""
    if not all(_finite_point(point) for point in (a, vertex, c)):
        return math.nan
    first, second = a - vertex, c - vertex
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-9:
        return math.nan
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _joint_flexion(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    value = _angle(a, vertex, c)
    return 180.0 - value if math.isfinite(value) else math.nan


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    result = values.astype(float).copy()
    half = window // 2
    for index in range(len(result)):
        segment = values[max(0, index - half) : min(len(values), index + half + 1)]
        finite = segment[np.isfinite(segment)]
        result[index] = np.median(finite) if finite.size else math.nan
    return result


def _find_peaks(values: np.ndarray, distance: int, prominence: float) -> list[tuple[int, float]]:
    """반복되는 보행 신호에서 간격과 돌출도를 만족하는 봉우리를 찾는다."""
    candidates: list[tuple[int, float]] = []
    radius = max(2, distance // 2)
    for index in range(1, len(values) - 1):
        if values[index] < values[index - 1] or values[index] <= values[index + 1]:
            continue
        left = values[max(0, index - radius) : index]
        right = values[index + 1 : min(len(values), index + radius + 1)]
        if not left.size or not right.size:
            continue
        local_prominence = values[index] - max(float(np.min(left)), float(np.min(right)))
        if local_prominence >= prominence:
            candidates.append((index, local_prominence))
    selected: list[tuple[int, float]] = []
    for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(candidate[0] - kept[0]) >= distance for kept in selected):
            selected.append(candidate)
    return sorted(selected)


def load_pose_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """주 러너의 좌표와 키포인트별 신뢰도를 배열로 불러온다."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    coordinates = np.full((len(frames), 26, 2), np.nan, dtype=float)
    confidence = np.zeros((len(frames), 26), dtype=float)
    for frame_index, frame in enumerate(frames):
        person = next(
            (item for item in frame.get("people", []) if item.get("track_id") == 0),
            None,
        )
        if person is None:
            continue
        raw = person.get("keypoints", [])
        scores = person.get("keypoint_scores", [])
        observed = person.get("observed", [True] * len(raw))
        imputed = person.get("imputed_keypoints", [None] * len(raw))
        for keypoint in range(min(26, len(raw))):
            point = raw[keypoint] if observed[keypoint] else imputed[keypoint]
            if point is not None:
                coordinates[frame_index, keypoint] = point
                confidence[frame_index, keypoint] = float(scores[keypoint])
    return coordinates, confidence


def _video_fps(details_path: Path, fallback: float) -> float:
    if not details_path.exists():
        return fallback
    details = json.loads(details_path.read_text(encoding="utf-8"))
    fps = float(details.get("video", {}).get("fps") or fallback)
    return fps if fps > 0 else fallback


def _point(data: np.ndarray, name: str) -> np.ndarray:
    return data[:, KEYPOINTS[name], :]


def _center(data: np.ndarray, first: str, second: str) -> np.ndarray:
    return np.nanmean(np.stack((_point(data, first), _point(data, second))), axis=0)


def infer_running_direction(data: np.ndarray) -> tuple[int, str]:
    """화면 오른쪽 진행은 +1, 왼쪽 진행은 -1로 반환한다."""
    nose_x = np.nanmedian(_point(data, "nose")[:, 0])
    neck_x = np.nanmedian(_point(data, "neck")[:, 0])
    if math.isfinite(nose_x) and math.isfinite(neck_x) and abs(nose_x - neck_x) > 2:
        return (1 if nose_x > neck_x else -1), "face_orientation"
    hip_x = _point(data, "hip_center")[:, 0]
    finite = hip_x[np.isfinite(hip_x)]
    if finite.size >= 2 and abs(finite[-1] - finite[0]) > 2:
        return (1 if finite[-1] > finite[0] else -1), "hip_motion"
    return 1, "default_screen_right"


def _body_scale(data: np.ndarray) -> float:
    shoulder = _center(data, "left_shoulder", "right_shoulder")
    hip = _point(data, "hip_center")
    sides = []
    for side in ("left", "right"):
        knee, ankle = _point(data, f"{side}_knee"), _point(data, f"{side}_ankle")
        sides.append(
            np.linalg.norm(shoulder - hip, axis=1)
            + np.linalg.norm(hip - knee, axis=1)
            + np.linalg.norm(knee - ankle, axis=1)
        )
    values = np.concatenate(sides)
    finite = values[np.isfinite(values) & (values > 1)]
    return float(np.median(finite)) if finite.size else 1.0


def detect_contacts(data: np.ndarray, fps: float, body_scale: float) -> list[Contact]:
    """발이 지면 근처 구간에 처음 진입한 프레임을 착지로 추정한다."""
    candidates: list[Contact] = []
    distance = max(2, round(fps * 0.35))
    minimum_descent = max(2.0, body_scale * 0.025)
    for side in ("left", "right"):
        foot_y = np.nanmax(
            np.stack((
                _point(data, f"{side}_heel")[:, 1],
                _point(data, f"{side}_big_toe")[:, 1],
                _point(data, f"{side}_ankle")[:, 1],
            )),
            axis=0,
        )
        foot_y = _smooth(foot_y, window=3)
        valid = np.isfinite(foot_y)
        if valid.sum() < 3:
            continue
        filled = np.interp(np.arange(len(foot_y)), np.flatnonzero(valid), foot_y[valid])
        ground_band = float(np.percentile(filled, 90)) - body_scale * 0.03
        near_ground = filled >= ground_band
        entries = np.flatnonzero(near_ground & ~np.r_[False, near_ground[:-1]])
        kept: list[Contact] = []
        for frame in entries:
            if frame == 0:
                continue
            prior = filled[max(0, frame - distance) : frame]
            descent = filled[frame] - float(np.min(prior)) if prior.size else 0.0
            if descent < minimum_descent:
                continue
            candidate = Contact(int(frame), side, descent)
            if kept and candidate.frame - kept[-1].frame < distance:
                if candidate.prominence > kept[-1].prominence:
                    kept[-1] = candidate
            else:
                kept.append(candidate)
        if len(kept) < 2:
            kept = [
                Contact(frame, side, value)
                for frame, value in _find_peaks(filled, distance, minimum_descent)
            ]
        candidates.extend(kept)
    candidates.sort(key=lambda item: item.frame)
    merged: list[Contact] = []
    collision = max(1, round(fps * 0.08))
    for candidate in candidates:
        if merged and candidate.frame - merged[-1].frame <= collision:
            if candidate.prominence > merged[-1].prominence:
                merged[-1] = candidate
        else:
            merged.append(candidate)
    if len(merged) < 3:
        return merged
    bouts: list[list[Contact]] = [[]]
    maximum_gap = fps * 1.5
    for contact in merged:
        if bouts[-1] and contact.frame - bouts[-1][-1].frame > maximum_gap:
            bouts.append([])
        bouts[-1].append(contact)
    return max(bouts, key=len)


def _side_points(data: np.ndarray, frame: int, side: str) -> dict[str, np.ndarray]:
    return {
        name: data[frame, KEYPOINTS[f"{side}_{name}"]]
        for name in (
            "shoulder", "elbow", "wrist", "hip", "knee", "ankle",
            "heel", "big_toe", "small_toe",
        )
    }


def _joint_series(data: np.ndarray, side: str, joint: str) -> np.ndarray:
    names = {
        "knee": ("hip", "knee", "ankle"),
        "elbow": ("shoulder", "elbow", "wrist"),
    }[joint]
    return np.asarray([
        _joint_flexion(
            data[index, KEYPOINTS[f"{side}_{names[0]}"]],
            data[index, KEYPOINTS[f"{side}_{names[1]}"]],
            data[index, KEYPOINTS[f"{side}_{names[2]}"]],
        )
        for index in range(len(data))
    ])


def _stance_end(data: np.ndarray, contact: Contact, next_same: int, body_scale: float) -> int:
    """착지 뒤 발이 연속으로 상승하기 시작한 프레임을 toe-off로 추정한다."""
    side = contact.side
    foot_y = np.nanmax(
        np.stack((
            _point(data, f"{side}_heel")[:, 1],
            _point(data, f"{side}_big_toe")[:, 1],
            _point(data, f"{side}_ankle")[:, 1],
        )),
        axis=0,
    )
    contact_y = foot_y[contact.frame]
    threshold = body_scale * 0.025
    start = min(next_same, contact.frame + 2)
    for frame in range(start, max(start, next_same - 1)):
        window = foot_y[frame : min(frame + 3, next_same)]
        if window.size >= 2 and np.isfinite(window).all() and np.all(window < contact_y - threshold):
            return frame
    return max(contact.frame + 1, min(next_same - 1, contact.frame + (next_same - contact.frame) // 2))


def _asymmetry(left: float, right: float) -> float:
    mean = (abs(left) + abs(right)) / 2.0
    return abs(left - right) / mean * 100.0 if mean > 1e-9 else math.nan


def _mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return round(float(np.mean(finite)), 2) if finite.size else None


def _feature(value, unit: str, status: str, reference: str, samples: int, *, method: str = "measured") -> dict:
    return {
        "value": value, "unit": unit, "status": status,
        "reference": reference, "sample_count": int(samples), "method": method,
    }


def _range_status(value: float | None, low: float, high: float) -> str:
    if value is None:
        return "insufficient_data"
    if value < low:
        return "below_reference"
    if value > high:
        return "above_reference"
    return "within_reference"


def extract_features(
    data: np.ndarray,
    confidence: np.ndarray,
    fps: float,
    direction: int | None = None,
) -> dict:
    if len(data) == 0:
        raise ValueError("pose_predictions.json에 프레임이 없습니다.")
    inferred_direction, direction_method = infer_running_direction(data)
    direction_overridden = direction is not None
    direction = direction or inferred_direction
    body_scale = _body_scale(data)
    contacts = detect_contacts(data, fps, body_scale)
    shoulder = _center(data, "left_shoulder", "right_shoulder")
    hip_center = _point(data, "hip_center")
    neck = _point(data, "neck")
    knee_series = {side: _joint_series(data, side, "knee") for side in ("left", "right")}
    elbow_series = {side: _joint_series(data, side, "elbow") for side in ("left", "right")}

    per_contact: list[dict] = []
    stance_times = {"left": [], "right": []}
    knee_excursions = {"left": [], "right": []}
    overstride = {"left": [], "right": []}
    pelvis_ankle = {"left": [], "right": []}
    for index, contact in enumerate(contacts):
        next_same = next(
            (item.frame for item in contacts[index + 1 :] if item.side == contact.side),
            len(data) - 1,
        )
        if next_same <= contact.frame + 2:
            continue
        stance_end = _stance_end(data, contact, next_same, body_scale)
        points = _side_points(data, contact.frame, contact.side)
        knee_ic = knee_series[contact.side][contact.frame]
        stance_knee = knee_series[contact.side][contact.frame : stance_end + 1]
        finite_knee = stance_knee[np.isfinite(stance_knee)]
        peak_knee = float(np.max(finite_knee)) if finite_knee.size else math.nan
        excursion = peak_knee - knee_ic if math.isfinite(peak_knee) and math.isfinite(knee_ic) else math.nan
        toe = np.nanmean(np.stack((points["big_toe"], points["small_toe"])), axis=0)
        foot_angle = (
            abs(math.degrees(math.atan2(toe[1] - points["heel"][1], toe[0] - points["heel"][0])))
            if _finite_point(toe) and _finite_point(points["heel"]) else math.nan
        )
        foot_angle = min(foot_angle, abs(180.0 - foot_angle)) if math.isfinite(foot_angle) else foot_angle
        tibia_vector = points["knee"] - points["ankle"]
        tibia = (
            abs(math.degrees(math.atan2(tibia_vector[0], -tibia_vector[1])))
            if _finite_point(tibia_vector) else math.nan
        )
        pelvis_distance = direction * (points["ankle"][0] - hip_center[contact.frame, 0])
        heel_distance = direction * (points["heel"][0] - hip_center[contact.frame, 0])
        normalized_overstride = heel_distance / body_scale
        overstride[contact.side].append(normalized_overstride)
        pelvis_ankle[contact.side].append(pelvis_distance / body_scale)
        stance_times[contact.side].append((stance_end - contact.frame) / fps)
        knee_excursions[contact.side].append(excursion)
        hip_extension_values = []
        late_stance_start = contact.frame + max(1, (stance_end - contact.frame) * 2 // 3)
        for frame in range(late_stance_start, stance_end + 1):
            hip = data[frame, KEYPOINTS[f"{contact.side}_hip"]]
            knee = data[frame, KEYPOINTS[f"{contact.side}_knee"]]
            side_shoulder = data[frame, KEYPOINTS[f"{contact.side}_shoulder"]]
            deviation = 180.0 - _angle(side_shoulder, hip, knee)
            if math.isfinite(deviation):
                sign = 1.0 if direction * (knee[0] - hip[0]) < 0 else -1.0
                hip_extension_values.append(sign * deviation)
        hip_extension = max(hip_extension_values) if hip_extension_values else math.nan
        per_contact.append({
            "frame": contact.frame + 1,
            "time_seconds": round(contact.frame / fps, 3),
            "side": contact.side,
            "estimated_toe_off_frame": stance_end + 1,
            "knee_flexion_at_ic_deg": round(knee_ic, 2) if math.isfinite(knee_ic) else None,
            "peak_knee_flexion_stance_deg": round(peak_knee, 2) if math.isfinite(peak_knee) else None,
            "knee_flexion_excursion_deg": round(excursion, 2) if math.isfinite(excursion) else None,
            "tibia_inclination_at_ic_deg": round(tibia, 2) if math.isfinite(tibia) else None,
            "foot_inclination_at_ic_deg": round(foot_angle, 2) if math.isfinite(foot_angle) else None,
            "hip_extension_late_stance_deg": round(hip_extension, 2) if math.isfinite(hip_extension) else None,
            "pelvis_to_ankle_ap_distance_px": round(float(pelvis_distance), 2),
            "pelvis_to_ankle_ap_distance_body_ratio": round(float(pelvis_distance / body_scale), 4),
            "heel_to_com_ap_distance_px": round(float(heel_distance), 2),
            "heel_to_com_ap_distance_body_ratio": round(float(normalized_overstride), 4),
        })

    contact_frames = np.asarray([item.frame for item in contacts], dtype=float)
    alternation_ratio = (
        sum(first.side != second.side for first, second in zip(contacts, contacts[1:]))
        / (len(contacts) - 1)
        if len(contacts) >= 2 else 0.0
    )
    cadence = step_asymmetry = None
    cadence_by_side = []
    for side in ("left", "right"):
        side_frames = np.asarray([item.frame for item in contacts if item.side == side], dtype=float)
        stride_intervals = np.diff(side_frames) / fps
        plausible = stride_intervals[(stride_intervals >= 0.4) & (stride_intervals <= 2.0)]
        if plausible.size:
            cadence_by_side.append(120.0 / float(np.median(plausible)))
    if cadence_by_side:
        cadence = float(np.mean(cadence_by_side))
    if len(contact_frames) >= 3 and alternation_ratio >= 0.75:
        intervals_by_side = {"left": [], "right": []}
        for previous, current in zip(contacts, contacts[1:]):
            if previous.side == current.side:
                continue
            intervals_by_side[current.side].append((current.frame - previous.frame) / fps)
        if all(intervals_by_side.values()):
            step_asymmetry = _asymmetry(
                float(np.mean(intervals_by_side["left"])),
                float(np.mean(intervals_by_side["right"])),
            )

    postural_lean_values, torso_values = [], []
    for item in per_contact:
        frame = ((item["frame"] - 1) + (item["estimated_toe_off_frame"] - 1)) // 2
        side = item["side"]
        ankle = data[frame, KEYPOINTS[f"{side}_ankle"]]
        if _finite_point(ankle) and _finite_point(shoulder[frame]):
            vector = shoulder[frame] - ankle
            postural_lean_values.append(math.degrees(math.atan2(direction * vector[0], -vector[1])))
        if _finite_point(neck[frame]) and _finite_point(hip_center[frame]):
            vector = neck[frame] - hip_center[frame]
            torso_values.append(math.degrees(math.atan2(direction * vector[0], -vector[1])))

    mean_stance = {side: _mean(values) for side, values in stance_times.items()}
    stance_asymmetry = (
        _asymmetry(mean_stance["left"], mean_stance["right"])
        if all(value is not None for value in mean_stance.values()) else None
    )
    mean_rom = {side: _mean(values) for side, values in knee_excursions.items()}
    knee_rom_asymmetry = (
        _asymmetry(mean_rom["left"], mean_rom["right"])
        if all(value is not None for value in mean_rom.values()) else None
    )
    mean_overstride = {side: _mean(values) for side, values in overstride.items()}
    overstride_asymmetry = (
        _asymmetry(mean_overstride["left"], mean_overstride["right"])
        if all(value is not None for value in mean_overstride.values()) else None
    )
    contact_values = lambda name: [item[name] for item in per_contact if item[name] is not None]
    knee_ic = _mean(contact_values("knee_flexion_at_ic_deg"))
    peak_knee = _mean(contact_values("peak_knee_flexion_stance_deg"))
    knee_excursion = _mean(contact_values("knee_flexion_excursion_deg"))
    tibia = _mean(contact_values("tibia_inclination_at_ic_deg"))
    foot = _mean(contact_values("foot_inclination_at_ic_deg"))
    hip_extension = _mean(contact_values("hip_extension_late_stance_deg"))
    lean, torso = _mean(postural_lean_values), _mean(torso_values)
    elbow_rom_values = []
    for side in ("left", "right"):
        finite = elbow_series[side][np.isfinite(elbow_series[side])]
        if finite.size:
            elbow_rom_values.append(float(np.percentile(finite, 95) - np.percentile(finite, 5)))
    elbow_rom = _mean(elbow_rom_values)
    observed_ratio = float(np.count_nonzero(confidence >= 0.5) / confidence.size) if confidence.size else 0.0

    features = {
        "postural_lean_deg": _feature(lean, "deg", _range_status(lean, 1.0, 11.5), "1.7° upright; 4.3° moderate; 6.1-11.5° large lean", len(postural_lean_values)),
        "torso_flexion_deg": _feature(torso, "deg", _range_status(torso, 7.0, 11.0), "7-11° is the document's recommended reference band", len(torso_values)),
        "knee_flexion_at_ic_deg": _feature(knee_ic, "deg", _range_status(knee_ic, 8.0, 17.2), "2-D studies: about 10.4-13.2°, SD-derived outer band about 8.0-17.2°", len(contact_values("knee_flexion_at_ic_deg"))),
        "peak_knee_flexion_stance_deg": _feature(peak_knee, "deg", _range_status(peak_knee, 40.0, 48.1), "about 45°; values below 40° were noted as low", len(contact_values("peak_knee_flexion_stance_deg"))),
        "knee_flexion_excursion_deg": _feature(knee_excursion, "deg", _range_status(knee_excursion, 20.77, 30.45), "20.77-30.45° reference interval", len(contact_values("knee_flexion_excursion_deg"))),
        "tibia_inclination_at_ic_deg": _feature(tibia, "deg", _range_status(tibia, 5.3, 11.7), "8.5±3.2° from vertical", len(contact_values("tibia_inclination_at_ic_deg"))),
        "foot_inclination_at_ic_deg": _feature(foot, "deg", _range_status(foot, 4.0, 16.4), "10.2±6.2°; document feedback band 4.0-16.4°", len(contact_values("foot_inclination_at_ic_deg"))),
        "hip_extension_late_stance_deg": _feature(hip_extension, "deg", _range_status(hip_extension, 10.0, 20.0), "The source notes use 10-20° as a coaching reference", len(contact_values("hip_extension_late_stance_deg")), method="estimated_from_2d_pose"),
        "cadence_spm": _feature(round(cadence, 2) if cadence is not None else None, "steps/min", _range_status(cadence, 164.0, 181.0), "preferred step rate 172.6±8.8 steps/min", len(contacts), method="estimated_from_stride_intervals"),
        "step_time_asymmetry_pct": _feature(round(step_asymmetry, 2) if step_asymmetry is not None else None, "%", _range_status(step_asymmetry, 0.0, 6.0), "about 2.1-2.2%; 6% is an approximate upper reference, not a diagnostic cutoff", max(0, len(contacts) - 1), method="estimated_from_contacts"),
        "estimated_stance_time_asymmetry_pct": _feature(round(stance_asymmetry, 2) if stance_asymmetry is not None else None, "%", _range_status(stance_asymmetry, 0.0, 5.0), "3.5-4.0% observed; 5% MVP reference", len(per_contact), method="estimated_from_2d_foot_motion"),
        "knee_rom_asymmetry_pct": _feature(round(knee_rom_asymmetry, 2) if knee_rom_asymmetry is not None else None, "%", _range_status(knee_rom_asymmetry, 0.0, 5.8), "3.8-5.8% natural asymmetry reported across age groups", len(per_contact)),
        "overstride_asymmetry_pct": _feature(round(overstride_asymmetry, 2) if overstride_asymmetry is not None else None, "%", "exploratory" if overstride_asymmetry is not None else "insufficient_data", "No diagnostic cutoff was established in the source notes", len(per_contact), method="body_scale_normalized"),
        "pelvis_to_ankle_ap_distance_body_ratio": _feature(_mean(value for values in pelvis_ankle.values() for value in values), "body_scale_ratio", "exploratory" if per_contact else "insufficient_data", "Uncalibrated horizontal distance; no validated ratio cutoff in the source notes", len(per_contact), method="body_scale_normalized"),
        "heel_to_com_ap_distance_body_ratio": _feature(_mean(value for values in overstride.values() for value in values), "body_scale_ratio", "exploratory" if per_contact else "insufficient_data", "The paper's metre threshold requires camera calibration; ratio is reported without clinical classification", len(per_contact), method="body_scale_normalized"),
        "elbow_angle_rom_deg": _feature(elbow_rom, "deg", "exploratory" if elbow_rom is not None else "insufficient_data", "Source notes did not provide a validated cutoff", len(elbow_rom_values)),
    }
    if len(contacts) < 3 or observed_ratio < 0.4:
        for name, feature in features.items():
            if name != "elbow_angle_rom_deg":
                feature["status"] = "insufficient_data"
    if alternation_ratio < 0.75:
        for name in (
            "step_time_asymmetry_pct",
            "estimated_stance_time_asymmetry_pct",
            "knee_rom_asymmetry_pct",
            "overstride_asymmetry_pct",
        ):
            features[name]["status"] = "insufficient_data"
    if len(contacts) >= 6 and observed_ratio >= 0.7 and alternation_ratio >= 0.75:
        analysis_quality = "good"
    elif len(contacts) >= 3 and observed_ratio >= 0.4:
        analysis_quality = "caution"
    else:
        analysis_quality = "insufficient"
    return {
        "schema_version": "2.0",
        "metadata": {
            "fps": round(float(fps), 3), "frames_analyzed": len(data),
            "running_direction": "screen_right" if direction == 1 else "screen_left",
            "running_direction_method": "cli_override" if direction_overridden else direction_method,
            "body_scale_px": round(body_scale, 2),
            "observed_keypoint_ratio": round(observed_ratio, 4),
            "contact_count": len(contacts),
            "contact_alternation_ratio": round(alternation_ratio, 4),
            "analysis_quality": analysis_quality,
        },
        "features": features,
        "contacts": per_contact,
        "limitations": [
            "Initial contact and toe-off are 2-D landmark estimates, not force-plate measurements.",
            "Real-world centimetres or metres cannot be recovered without camera calibration or a known scale.",
            "Reference bands describe study samples and are not medical diagnostic thresholds.",
            "Side-view, fixed-camera video with the full body and several strides is required.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("poc_folder", type=Path)
    parser.add_argument("run_folder")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback FPS when details.json is absent")
    parser.add_argument("--running-direction", choices=("auto", "left", "right"), default="auto")
    args = parser.parse_args()
    output_dir = args.poc_folder / "run" / args.run_folder / "outputs"
    pose_path = output_dir / "pose_predictions.json"
    if not pose_path.exists():
        raise FileNotFoundError(f"포즈 결과가 없습니다: {pose_path}")
    data, confidence = load_pose_data(pose_path)
    fps = _video_fps(output_dir / "details.json", args.fps)
    direction = {"left": -1, "right": 1}.get(args.running_direction)
    result = extract_features(data, confidence, fps, direction)
    feature_path = output_dir / "feature_results.json"
    feature_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"피처 {len(result['features'])}개 저장 완료: {feature_path}")
    print(f"착지 이벤트: {result['metadata']['contact_count']}개")


if __name__ == "__main__":
    main()
