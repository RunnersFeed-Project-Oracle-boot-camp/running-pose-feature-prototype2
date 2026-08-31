"""API를 사용할 수 없을 때 같은 입력에 같은 결과를 만드는 로컬 리포트."""

from __future__ import annotations


LABELS = {
    "postural_lean_deg": "전신 전방 기울기",
    "torso_flexion_deg": "몸통 전방 기울기",
    "knee_flexion_at_ic_deg": "착지 시 무릎 굴곡",
    "peak_knee_flexion_stance_deg": "입각기 최대 무릎 굴곡",
    "knee_flexion_excursion_deg": "착지 후 무릎 굴곡 변화량",
    "tibia_inclination_at_ic_deg": "착지 시 정강이 기울기",
    "foot_inclination_at_ic_deg": "착지 시 발 기울기",
    "hip_extension_late_stance_deg": "후기 입각기 고관절 신전",
    "cadence_spm": "분당 발 착지 횟수",
    "step_time_asymmetry_pct": "좌우 스텝 시간 차이",
    "estimated_stance_time_asymmetry_pct": "좌우 지면 체류 시간 차이",
    "knee_rom_asymmetry_pct": "좌우 무릎 가동범위 차이",
}

ADVICE = {
    "postural_lean_deg": "허리만 접기보다 몸을 길게 세운 채 발목에서 가볍게 기울이는 느낌을 확인해 보세요.",
    "torso_flexion_deg": "시선을 앞에 두고 가슴과 골반 사이를 길게 유지해 몸통 굽힘을 조금 줄여보세요.",
    "knee_flexion_at_ic_deg": "발을 몸에서 멀리 내밀지 말고 몸 아래에 가깝게 가볍게 착지해 보세요.",
    "peak_knee_flexion_stance_deg": "착지 후 무릎을 잠그지 말고 자연스럽게 충격을 흡수하는 느낌을 유지하세요.",
    "knee_flexion_excursion_deg": "착지 직후 무릎이 자연스럽게 더 굽혀지도록 보폭과 힘을 조금 낮춰보세요.",
    "tibia_inclination_at_ic_deg": "정강이를 과하게 앞으로 뻗지 않도록 발을 골반 아래에 가깝게 내려놓아 보세요.",
    "foot_inclination_at_ic_deg": "발 착지 형태를 억지로 바꾸기보다 발이 몸 앞쪽으로 멀리 나가는지 함께 확인하세요.",
    "hip_extension_late_stance_deg": "뒤로 강하게 차기보다 엉덩이 뒤쪽으로 다리가 자연스럽게 지나가게 해보세요.",
    "cadence_spm": "현재 리듬에서 보폭을 조금 줄이고 분당 발걸음을 약 5%씩 점진적으로 높여보세요.",
    "step_time_asymmetry_pct": "양발을 같은 박자로 가볍게 내려놓는 느낌에 집중해 보세요.",
    "estimated_stance_time_asymmetry_pct": "한쪽 발에 오래 머물지 않도록 좌우 착지와 도약 리듬을 맞춰보세요.",
    "knee_rom_asymmetry_pct": "양쪽 무릎을 들어 올리고 펴는 크기를 비슷하게 맞추는 느낌으로 달려보세요.",
}

PRIORITY = (
    "knee_flexion_excursion_deg",
    "knee_rom_asymmetry_pct",
    "estimated_stance_time_asymmetry_pct",
    "knee_flexion_at_ic_deg",
    "tibia_inclination_at_ic_deg",
    "torso_flexion_deg",
    "foot_inclination_at_ic_deg",
    "postural_lean_deg",
    "cadence_spm",
    "hip_extension_late_stance_deg",
)


def _measurement(name: str, feature: dict) -> str:
    value = feature.get("value")
    unit = feature.get("unit", "")
    return f"{LABELS.get(name, name)} {value}{unit}"


def build_local_report(result: dict, reason: str = "offline") -> str:
    metadata = result.get("metadata", {})
    features = result.get("features", {})
    usable = {
        name: feature
        for name, feature in features.items()
        if feature.get("value") is not None
        and feature.get("status") not in {"insufficient_data", "exploratory"}
    }
    good = [
        (name, feature)
        for name, feature in usable.items()
        if feature.get("status") == "within_reference"
    ]
    needs_attention = [
        (name, usable[name])
        for name in PRIORITY
        if name in usable and usable[name].get("status") in {"below_reference", "above_reference"}
    ]

    quality = metadata.get("analysis_quality", "unknown")
    contacts = metadata.get("contact_count", 0)
    observed = metadata.get("observed_keypoint_ratio")
    observed_text = f", 키포인트 관측률 {observed * 100:.1f}%" if isinstance(observed, (int, float)) else ""
    if needs_attention:
        summary = f"측정 품질은 {quality}이며, 우선 조정해 볼 항목은 {LABELS.get(needs_attention[0][0], needs_attention[0][0])}입니다."
    elif good:
        summary = f"측정 품질은 {quality}이며, 주요 측정값이 연구 참고 구간과 유사합니다."
    else:
        summary = "판정 가능한 측정값이 부족해 촬영 조건을 먼저 점검해야 합니다."

    lines = [
        "# 러닝 자세 분석 리포트",
        "",
        f"> 생성 방식: 로컬 규칙 기반 분석 ({reason}). 연구 참고 구간은 진단 기준이 아닙니다.",
        "",
        "## 한줄 요약",
        "",
        summary,
        "",
        "## 잘 유지할 점",
        "",
    ]
    if good:
        for name, feature in good[:2]:
            lines.append(f"- {_measurement(name, feature)}: 연구 표본의 참고 구간과 유사합니다.")
    else:
        lines.append("- 현재 데이터에서는 참고 구간 안으로 분류된 핵심 항목이 충분하지 않습니다.")

    lines.extend(["", "## 조정해 볼 점", ""])
    if needs_attention:
        for name, feature in needs_attention[:2]:
            direction = "참고 구간보다 낮게" if feature.get("status") == "below_reference" else "참고 구간보다 높게"
            lines.append(f"- {_measurement(name, feature)}: {direction} 관찰됐습니다. {ADVICE.get(name, '')}")
        cadence = features.get("cadence_spm", {})
        if cadence.get("value") and any(name == "cadence_spm" for name, _ in needs_attention[:2]):
            value = float(cadence["value"])
            lines.append(f"  - 급격히 바꾸지 말고 우선 약 {value * 1.05:.0f}-{value * 1.10:.0f} steps/min 범위에서 편안함을 확인하세요.")
    else:
        lines.append("- 현재 리듬과 자세를 유지하면서 동일한 조건에서 반복 측정해 보세요.")

    lines.extend([
        "",
        "## 측정 한계",
        "",
        f"- 분석에 사용된 착지 이벤트는 {contacts}개{observed_text}입니다.",
        "- Initial Contact와 toe-off는 2D 포즈 좌표 기반 추정이며 포스 플레이트 측정이 아닙니다.",
        "- 카메라 보정이 없어 실제 cm 또는 m 단위 거리와 직접 비교하지 않았습니다.",
        "- 통증이나 불편함이 있으면 이 결과만으로 판단하지 말고 전문가와 상담하세요.",
        "",
    ])
    return "\n".join(lines)
