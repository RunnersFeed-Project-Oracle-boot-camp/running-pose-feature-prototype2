from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from scripts.Agent.local_report import build_local_report
from scripts.Agent.prompts import INPUT_DATA, INSTRUCTION, PERSONA
from scripts.features.papers import PAPER_EVIDENCE


def generate_report(features: dict, api_key: str | None) -> tuple[str, str]:
    if not api_key:
        return build_local_report(features, "API key 없음"), "local"

    prompt = ChatPromptTemplate.from_messages([
        ("system", PERSONA + "\n" + INSTRUCTION),
        ("human", INPUT_DATA),
    ])
    model = ChatOpenAI(model="gpt-5-nano", temperature=0, api_key=api_key)
    chain = prompt | model | StrOutputParser()
    try:
        report = chain.invoke({
            "features": json.dumps(features, ensure_ascii=False, indent=2),
            "paper_evidence": json.dumps(PAPER_EVIDENCE, ensure_ascii=False, indent=2),
        })
        return report, "openai"
    except Exception as error:
        reason = f"OpenAI API 사용 불가: {type(error).__name__}"
        print(f"{reason}; 로컬 규칙형 리포트로 전환합니다.")
        return build_local_report(features, reason), "local"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("run_folder")
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    output_dir = project_dir / "PoC" / "run" / args.run_folder / "outputs"
    features_path = output_dir / "feature_results.json"
    if not features_path.exists():
        raise FileNotFoundError(f"피처 결과가 없습니다: {features_path}")

    load_dotenv(project_dir / ".env")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("_OPENAI_API_KEY")
    features = json.loads(features_path.read_text(encoding="utf-8"))
    report, source = generate_report(features, api_key)
    report_path = output_dir / "running_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n리포트 저장 완료 ({source}): {report_path}")


if __name__ == "__main__":
    main()
