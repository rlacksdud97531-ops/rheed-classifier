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
    st.subheader("Decision rule")
    st.markdown(
        "1. **Streaks with kikuchi line** — (0.75 < Streaks < 0.917, 0.0677 < Mixed ≤ 0.2, Spotty ≤ 0.66) "
        "or (0.55 < Streaks < 0.56, 0.25 < Mixed < 0.27, 0.16 < Spotty < 0.18) "
        "or (0.954 < Streaks < 0.956, 0.298 < Mixed < 0.3, 0.0151 < Spotty < 0.0153)\n"
        "2. **Mixed · Streak-dominant** — Streaks ≥ 0.75, Mixed ≥ 0.0893, Spotty ≤ 0.069\n"
        "3. **Mixed · Spotty-dominant** — Streaks ≤ 0.7191, Mixed ≥ 0.1827, Spotty > 0.048\n"
        "4. **Spotty** — Spotty ≥ 0.55\n"
        "5. **Streaks** — Streaks > 0.8952\n"
        "6. **Unclear** — none of the above"
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
    # V2 줌 크롭 (band_h=0.60: 전체 패턴을 담아야 Mixed/Spotty 증거가 살아있음)
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
        p_streak = probs.get('Streaks', 0.0)
        p_spotty = probs.get('Spotty', 0.0)
        p_mixed  = probs.get('Mixed', 0.0)
        # 결정 규칙 (데이터 기반 경계):
        #  1) Streaks with kikuchi line  : (0.75 < Streaks < 0.917, 0.0677 < Mixed <= 0.2, Spotty <= 0.66)
        #                                   OR (0.55 < Streaks < 0.56, 0.25 < Mixed < 0.27, 0.16 < Spotty < 0.18)
        #                                   OR (0.954 < Streaks < 0.956, 0.298 < Mixed < 0.3, 0.0151 < Spotty < 0.0153)
        #  2) Mixed · Streak-dominant    : Streaks >= 0.75,   Mixed >= 0.0893, Spotty <= 0.069
        #  3) Mixed · Spotty-dominant    : Streaks <= 0.7191, Mixed >= 0.1827, Spotty > 0.048
        #  4) Spotty                     : Spotty  >= 0.55
        #  5) Streaks                    : Streaks > 0.8952
        #  6) 어느 것도 아니면 Unclear
        if ((0.75 < p_streak < 0.917 and 0.0677 < p_mixed <= 0.2 and p_spotty <= 0.66)
                or (0.55 < p_streak < 0.56 and 0.25 < p_mixed < 0.27 and 0.16 < p_spotty < 0.18)
                or (0.954 < p_streak < 0.956 and 0.298 < p_mixed < 0.3 and 0.0151 < p_spotty < 0.0153)):
            label = "Streaks with kikuchi line"
        elif p_streak >= 0.75 and p_mixed >= 0.0893 and p_spotty <= 0.069:
            label = "Mixed · Streak-dominant"
        elif p_streak <= 0.7191 and p_mixed >= 0.1827 and p_spotty > 0.048:
            label = "Mixed · Spotty-dominant"
        elif p_spotty >= 0.55:
            label = "Spotty"
        elif p_streak > 0.8952:
            label = "Streaks"
        else:
            label = "Unclear"
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
