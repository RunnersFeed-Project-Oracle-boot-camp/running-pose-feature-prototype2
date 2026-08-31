import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from rtmlib import draw_skeleton


def render(img_dir: Path, output_dir: Path) -> None:
    RENDER_PATH = output_dir / "rendered"
    with open(output_dir / "pose_predictions.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    
    RENDER_PATH.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(img_dir.glob("*.png"))

    # 자세 데이터는 유효 구간만 가지므로 원본 파일명으로 두 시퀀스를 맞춘다.
    j = 0
    for i, image_path in enumerate(tqdm(image_paths, desc="Rendering")):
        img = cv2.imread(image_path)

        if j == len(data["frames"]):
            cv2.imwrite(RENDER_PATH / f"{i+1:08d}.png", img)
            continue

        if data["frames"][j]["image_path"].split('/')[-1] != str(image_path).split('/')[-1]:
            cv2.imwrite(RENDER_PATH / f"{i+1:08d}.png", img)
            continue

        user = data["frames"][j]["people"][0]
        j += 1
    
        keypoints = np.asarray(user["keypoints"], dtype=np.float32)[np.newaxis, :]
        scores = np.asarray(user['keypoint_scores'], dtype=np.float32)[np.newaxis, :]

        result = draw_skeleton(
            img.copy(),
            keypoints,
            scores,
            openpose_skeleton=False,
            kpt_thr=0.3,
            radius=4,
            line_width=2,
        )
        cv2.imwrite(RENDER_PATH / f"{i+1:08d}.png", result)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path)
    parser.add_argument("hpe_data", type=Path)
    args = parser.parse_args()

    start = time.perf_counter()
    render(args.frames, args.hpe_data)
    end = time.perf_counter()

    print(f"이미지 렌더링 완료.\n소요 시간 : {end - start:.2f}s")
