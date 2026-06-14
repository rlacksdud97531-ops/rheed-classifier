"""
RHEED 중심 peak(specular) 검출 데모 (Streamlit)
실행:  streamlit run app_peak.py
필요 파일: rheed_peak.py, rheed_crop.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import streamlit as st
import rheed_peak as RP

st.set_page_config(page_title="RHEED Peak Finder", page_icon="🎯", layout="wide")
st.markdown("<style>[data-testid='stFileUploaderDropzoneInstructions']{display:none !important;}</style>",
            unsafe_allow_html=True)
st.title("RHEED Center Peak Finder")

with st.sidebar:
    st.caption("가장 밝은 중심 peak(specular)을 자동 검출 → 십자(빨강) + ROI 박스(초록)")
    smooth = st.slider("smoothing σ", 1.0, 8.0, 3.0, 0.5)
    win = st.slider("fit window (px)", 8, 40, 20, 2)
    roi_k = st.slider("ROI size (×σ)", 2, 10, 5, 1)
    skip_top = st.checkbox("shadow edge 위 무시", value=True,
                           help="raw 원형 이미지의 위쪽 반사 artifact 회피")

up = st.file_uploader("RHEED image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
                      label_visibility="collapsed")
if up is None:
    st.info("RHEED 이미지를 올리면 중심 peak를 검출합니다.")
    st.stop()

try:
    g = tf.io.decode_image(up.read(), channels=1, expand_animations=False).numpy().squeeze().astype(np.float32)
except Exception as ex:
    st.error(f"이미지를 읽을 수 없습니다: {ex}")
    st.stop()

res = RP.find_center_peak(g, smooth=smooth, win=win, skip_top=skip_top)
vis = RP.draw_peak(g, res, roi_k=roi_k)

c1, c2 = st.columns([3, 1])
with c1:
    st.image(vis, caption="detected peak (red) + ROI (green)", use_container_width=True)
with c2:
    st.metric("center x", f"{res['x']:.1f} px")
    st.metric("center y", f"{res['y']:.1f} px")
    st.metric("sigma", f"{res['sigma']:.1f} px")
    st.caption(f"method: **{res['method']}**")
    st.caption(f"image: {g.shape[1]}×{g.shape[0]}")
