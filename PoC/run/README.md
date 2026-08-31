아래 구조와 같이 각 테스트 영상별로 입력 영상부터 출력까지의 산출물들을 하나의 폴더에서 관리하고자 한다.
```text
runs/
└── test1/
    ├── input.mp4 
    ├── inputs/
    │   ├── 00000001.png
    │   └── 00000500.png
    └── outputs/
        ├── pose_predictions.json
        ├── details.json
        ├── feature_results.json
        ├── rendered/
        │   ├── 00000001.png
        │   └── 00000500.png
        ├── rendered.mp4
        └── running_report.md
```
