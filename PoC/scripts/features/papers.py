"""코칭 리포트에 전달하는 지표별 연구 근거와 참고 범위.

참고 범위는 정리된 연구 표본에서 관찰된 값이며 보편적인 진단 기준이 아니다.
"""

PAPER_EVIDENCE = {
    "posture": {
        "postural_lean_deg": "1.7±0.7° upright, 4.3±0.8° moderate, 6.1-11.5° large forward lean.",
        "torso_flexion_deg": "The source notes use 7-11° as a coaching reference; 15°+ needs caution.",
    },
    "landing_and_impact": {
        "knee_flexion_at_ic_deg": "Reported 2-D means include 10.43° and 12.70-13.2°; interpret roughly 8.0-17.2° as a sample-derived outer band.",
        "peak_knee_flexion_stance_deg": "Healthy running commonly approaches 45°; below 40° was noted as relatively stiff.",
        "knee_flexion_excursion_deg": "A reported healthy-runner interval was 20.77-30.45°.",
        "tibia_inclination_at_ic_deg": "One adolescent sample reported 8.5±3.2° from vertical.",
        "foot_inclination_at_ic_deg": "One adolescent sample reported 10.2±6.2°; the notes use 4.0-16.4° as a feedback band.",
        "hip_extension_late_stance_deg": "The source notes use 10-20° as a coaching reference; the POC value is estimated from 2-D pose.",
        "pelvis_to_ankle_ap_distance_body_ratio": "No calibrated distance or validated normalized cutoff is available, so this is exploratory only.",
        "heel_to_com_ap_distance_body_ratio": "The cited 0.102±0.030 m cannot be compared to an uncalibrated pixel video. Treat the normalized ratio as exploratory only.",
    },
    "rhythm_and_symmetry": {
        "cadence_spm": "Preferred step rate was 172.6±8.8 steps/min in one study. A personal 5-10% adjustment is more appropriate than a universal target.",
        "step_time_asymmetry_pct": "Healthy groups averaged about 2.1-2.2%; 6% is an approximate upper reference, not a diagnostic cutoff.",
        "estimated_stance_time_asymmetry_pct": "Groups averaged about 3.5-4.0%; 5% is an MVP reference and the POC value is a 2-D estimate.",
        "knee_rom_asymmetry_pct": "Age-group means were about 3.8-5.8%; repeated values around 10%+ may merit a gentle balance cue, not a diagnosis.",
    },
    "rules": [
        "Use only measured values whose status is not insufficient_data.",
        "Call exploratory and estimated measurements by those names.",
        "Do not translate association into a guaranteed injury reduction or medical conclusion.",
        "Prefer one or two high-confidence, actionable cues over commenting on every metric.",
    ],
}
