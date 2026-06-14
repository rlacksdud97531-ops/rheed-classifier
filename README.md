# RHEED Pattern Classifier

RHEED 회절 패턴을 **Streaks / Spotty / Mixed** 3-class로 분류하는 CNN + 실시간 데모.
전이학습(EfficientNetV2B0) · growth 단위 group-CV(누수 없음) · 16-bit 이미지 처리 · Safety-First 추론 정책.

## 파일
| 파일 | 설명 |
|---|---|
| `rheed_cv.py` | 5-fold StratifiedGroupKFold 교차검증 + OOF + Unclear/Safety-First 정책 시뮬레이션 |
| `rheed_deploy.py` | 전체 데이터로 최종 모델 학습(`--train`) + `predict_frame()` + `TemporalSmoother` |
| `rheed_crop.py` | raw 이미지 자동 크롭 — `pattern_box`(테두리 제거, V1) / `zoom_box`(줄무늬·스팟 확대, V2) |
| `app.py` | Streamlit 데모 — 이미지 업로드 → 크롭 → 분류 결과 |

## 데이터·모델 (repo 제외)
용량/저작권 문제로 `.gitignore` 처리됨: `balanced_dataset/`, `zoom_dataset/`, `results/*.keras`, 논문 PDF 등.
실행하려면 별도로 데이터셋과 학습된 모델(`results/rheed_final.keras`)이 필요합니다.

## 실행
```bash
pip install -r requirements.txt

# 1) 교차검증 (성능 추정)
python rheed_cv.py                                  # V1 (balanced_dataset)
RHEED_DATA=zoom_dataset python rheed_cv.py          # V2 (zoom_dataset)

# 2) 최종 모델 학습 (전체 데이터)
python rheed_deploy.py --train

# 3) Streamlit 데모
streamlit run app.py
```

## 핵심 메모
- **이미지가 16-bit PNG** → PIL은 흰색으로 포화시킴. 반드시 `tf.io.decode_image` 사용.
- **growth 단위 분리** → 같은 성장의 인접 프레임이 train/test에 섞이면 정확도가 부풀려짐.
- 정직한 group-CV 기준 **macro-F1 ≈ 0.79** (단일 분할의 낙관적 값 아님).
- GPU는 네이티브 Windows에서 불가(TF≥2.11) → WSL2 또는 Colab.

> 학부 졸업 프로젝트. 데이터: IV-VI 칼코게나이드 박막/양자점 MBE 성장의 RHEED 패턴.
