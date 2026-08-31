import math
import unittest

import numpy as np

from scripts.features.feature_extract import (
    KEYPOINTS,
    _angle,
    _find_peaks,
    extract_features,
)
from scripts.Agent.local_report import build_local_report


def synthetic_run(frames: int = 180) -> tuple[np.ndarray, np.ndarray]:
    data = np.full((frames, 26, 2), np.nan, dtype=float)
    confidence = np.ones((frames, 26), dtype=float)
    for frame in range(frames):
        phase = 2 * math.pi * frame / 20
        data[frame, KEYPOINTS["nose"]] = [112, 58]
        data[frame, KEYPOINTS["neck"]] = [100, 80]
        data[frame, KEYPOINTS["hip_center"]] = [92, 165]
        for side, offset, phase_shift in (("left", -3, 0), ("right", 3, math.pi)):
            cycle = phase + phase_shift
            ankle_y = 285 + 24 * math.cos(cycle)
            ankle_x = 102 + offset + 9 * math.sin(cycle)
            data[frame, KEYPOINTS[f"{side}_shoulder"]] = [100 + offset, 82]
            data[frame, KEYPOINTS[f"{side}_hip"]] = [92 + offset, 165]
            data[frame, KEYPOINTS[f"{side}_knee"]] = [97 + offset + 5 * math.sin(cycle), 225]
            data[frame, KEYPOINTS[f"{side}_ankle"]] = [ankle_x, ankle_y]
            data[frame, KEYPOINTS[f"{side}_heel"]] = [ankle_x - 5, ankle_y + 2]
            data[frame, KEYPOINTS[f"{side}_big_toe"]] = [ankle_x + 18, ankle_y]
            data[frame, KEYPOINTS[f"{side}_small_toe"]] = [ankle_x + 17, ankle_y - 1]
            data[frame, KEYPOINTS[f"{side}_elbow"]] = [82 + offset, 120 + 8 * math.sin(cycle)]
            data[frame, KEYPOINTS[f"{side}_wrist"]] = [108 + offset, 145 + 14 * math.sin(cycle)]
    return data, confidence


class GeometryTests(unittest.TestCase):
    def test_angle(self):
        value = _angle(np.array([0.0, -1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        self.assertAlmostEqual(value, 90.0)

    def test_peak_distance_and_prominence(self):
        values = np.array([0, 1, 6, 1, 0, 1, 5, 1, 0], dtype=float)
        self.assertEqual([index for index, _ in _find_peaks(values, 3, 3)], [2, 6])


class FeaturePipelineTests(unittest.TestCase):
    def test_extracts_multi_feature_schema(self):
        data, confidence = synthetic_run()
        result = extract_features(data, confidence, fps=30.0, direction=1)
        self.assertEqual(result["schema_version"], "2.0")
        self.assertGreaterEqual(result["metadata"]["contact_count"], 12)
        self.assertEqual(result["metadata"]["analysis_quality"], "good")
        self.assertAlmostEqual(result["features"]["cadence_spm"]["value"], 180.0, delta=3.0)
        self.assertIn("knee_flexion_at_ic_deg", result["features"])
        self.assertIn("knee_rom_asymmetry_pct", result["features"])
        self.assertEqual(
            result["features"]["heel_to_com_ap_distance_body_ratio"]["status"],
            "exploratory",
        )
        self.assertTrue(result["contacts"])

    def test_empty_frames_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "프레임"):
            extract_features(np.empty((0, 26, 2)), np.empty((0, 26)), fps=30.0)

    def test_short_clip_is_marked_insufficient(self):
        data, confidence = synthetic_run(frames=18)
        result = extract_features(data, confidence, fps=30.0, direction=1)
        self.assertEqual(result["metadata"]["analysis_quality"], "insufficient")
        self.assertEqual(result["features"]["cadence_spm"]["status"], "insufficient_data")

    def test_local_report_is_available_without_api(self):
        data, confidence = synthetic_run()
        result = extract_features(data, confidence, fps=30.0, direction=1)
        report = build_local_report(result, "test")
        self.assertIn("# 러닝 자세 분석 리포트", report)
        self.assertIn("## 측정 한계", report)


if __name__ == "__main__":
    unittest.main()
