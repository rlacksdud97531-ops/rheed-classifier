"""
RHEED 단일 이미지 분류 데모 (Streamlit)
실행:  streamlit run rheed_app.py
사전조건:  results/rheed_final.keras  (python rheed_deploy.py --train 으로 생성)
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import rheed_deploy as D
import rheed_crop as RC
import rheed_peak as RP

st.set_page_config(page_title="RHEED 분류기", page_icon="🔬", layout="wide")

# 라벨 -> (이모지, streamlit 박스 종류, 설명)
STATUS = {
    'Streaks': ('✅', 'success', '매끈한 2D 층상 성장 (FM) — 정상'),
    'Spotty':  ('✅', 'success', '양자점 / 3D 섬 (VW) — 정상'),
    'Mixed':   ('⚠️', 'error',   '전이 / 거칠어진 패턴 — 확인 필요'),
    'Unclear': ('❓', 'warning', '모델 확신 부족 — 사람이 확인'),
}


@st.cache_resource
def get_model():
    return D.load_classifier()


st.title("RHEED Image Analysis")
# 파일 업로더의 "200MB per file • PNG, JPG" 안내 블록 숨기기
st.markdown(
    "<style>[data-testid='stFileUploaderDropzoneInstructions']{display:none !important;}</style>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Classes")
    st.markdown(
        "- **Streaks** — smooth 2D layer-by-layer film (Frank–van der Merwe)\n"
        "- **Spotty** — 3D islands / quantum dots (Volmer–Weber)\n"
        "- **Mixed** — rough or transitional growth — needs attention"
    )
    st.divider()
    st.markdown("📧 [rlacksdud97531@gmail.com](mailto:rlacksdud97531@gmail.com)")

# 모델 로드 (없으면 안내)
try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

up = st.file_uploader("RHEED 이미지", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
if up is None:
    st.stop()

# 디코드 (16-bit 안전) -> uint8 [0,255]
try:
    img = tf.io.decode_image(up.read(), channels=3, expand_animations=False).numpy().astype('uint8')
except Exception as ex:
    st.error(f"이미지를 읽을 수 없습니다: {ex}")
    st.stop()

tab_cls, tab_peak = st.tabs(["🔬 Classification", "🎯 Center peak"])

with tab_cls:
    # V2 줌 크롭 (항상 적용)
    t, b, l, r = RC.zoom_box(img)
    fed = img[t:b, l:r]

    # 원본에 V2 줌 박스(빨강) 표시
    overlay = img.copy()
    for (y0, y1, x0, x1) in [(t, t + 3, l, r), (b - 3, b, l, r), (t, b, l, l + 3), (t, b, r - 3, r)]:
        overlay[max(0, y0):max(0, y1), max(0, x0):max(0, x1), 0] = 255
        overlay[max(0, y0):max(0, y1), max(0, x0):max(0, x1), 1] = 0
        overlay[max(0, y0):max(0, y1), max(0, x0):max(0, x1), 2] = 0

    res = D.predict_frame(model, fed)

    c1, c2, c3 = st.columns([1.1, 1.1, 1])
    with c1:
        st.image(overlay, use_container_width=True)
    with c2:
        st.image(fed, use_container_width=True)
    with c3:
        probs = res['probs']
        top = max(probs, key=probs.get)                       # 확률 최고 클래스
        # Safety-First: Mixed(전이/거칠어진 성장)는 놓치면 손해 -> Streaks/Spotty가
        # 기준점(0.55)을 넘어도 Mixed가 0.20을 넘으면 Mixed로 분류한다.
        if probs.get('Mixed', 0.0) > 0.20:
            label = "Mixed"
        elif probs[top] > 0.55:                               # 그 외엔 top이 0.55 넘으면 그 클래스
            label = top
        else:
            label = "Unclear"                                 # 아무것도 확실치 않으면 Unclear
        st.subheader(label)
        st.markdown("**Class probabilities**")
        st.bar_chart(pd.DataFrame({'확률': probs}))

with tab_peak:
    # streak 대칭성으로 중앙(00) streak를 자동 검출 -> 슬라이더로 보정 (반자동)
    H, W = img.shape[:2]
    auto = RP.find_center_peak(img)
    st.caption("Auto-detected from streak symmetry (middle streak = center). Adjust with sliders if needed.")
    s1, s2, s3 = st.columns(3)
    cx = s1.slider("center x", 0, W, int(np.clip(auto['x'], 0, W)))
    cy = s2.slider("center y", 0, H, int(np.clip(auto['y'], 0, H)))
    roi = s3.slider("ROI radius (px)", 10, 300, int(np.clip(5 * auto['sigma'], 20, 300)))
    vis = RP.draw_peak(img, {'x': cx, 'y': cy, 'sigma': roi, 'streaks_x': auto['streaks_x']},
                       roi_k=1, show_streaks=True)
    st.image(vis, use_container_width=True)
    info = f"center = ({cx}, {cy}) · ROI ±{roi}px · {auto['n_streaks']} streaks detected"
    if auto['n_streaks'] >= 2 and np.isfinite(auto['spacing']):
        info += f" · mean spacing ≈ {auto['spacing']:.0f} px"
    st.caption(info)
