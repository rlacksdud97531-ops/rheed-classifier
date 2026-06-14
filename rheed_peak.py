"""
rheed_peak.py — RHEED 가장 밝은 중심 peak(specular spot) 자동 검출.
  smooth -> 2D argmax(거친 중심) -> window 내 2D Gaussian fit(sub-pixel 중심+폭).
  포화/실패 시 intensity-weighted centroid로 fallback.
검출된 (x, y, sigma)를 ROI 중심으로 써서 reconstruction / lattice constant 분석에 사용.
"""
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter, median_filter, white_tophat
from scipy.optimize import curve_fit


def load_gray(path):
    """16-bit 안전 로드 -> 2D float32 그레이스케일."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=1, expand_animations=False)
    return img.numpy().squeeze().astype(np.float32)


def _gauss2d(coords, A, xc, yc, sx, sy, off):
    x, y = coords
    return (A * np.exp(-((x - xc) ** 2 / (2 * sx ** 2) + (y - yc) ** 2 / (2 * sy ** 2))) + off).ravel()


def find_center_peak(img, smooth=3.0, win=20, search_region=None, skip_top=True):
    """가장 밝은 peak의 sub-pixel 중심 + 폭.
       img: 2D 또는 (H,W,3). search_region=(y0,y1,x0,x1)로 검색범위 제한 가능(엉뚱한 spot 방지).
       skip_top=True: shadow edge 위(어두운 영역의 반사 artifact)는 검색 제외 (raw 원형 이미지용).
       반환 dict: x, y, sigma, method, coarse, A, offset."""
    g = img.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    H, W = g.shape
    base = median_filter(g, size=3)                          # hot pixel 제거
    # white top-hat: 넓고 흐린 glow를 빼고 컴팩트한 spot만 남김 -> argmax가 glow 대신 진짜 spot을 잡음
    sm = gaussian_filter(white_tophat(base, size=max(15, int(0.025 * max(H, W)))), smooth)

    if search_region is None and skip_top:
        # specular는 shadow edge 바로 아래 + 중앙. 아래쪽 깊은 glow / 좌우·하단 원형 테두리는 제외.
        try:
            import rheed_crop
            edge = int(rheed_crop.find_crop_row(g))
        except Exception:
            edge = 0
        y0 = min(H - 1, edge + int(0.03 * H))                # shadow edge 살짝 아래 (edge artifact 회피)
        y1 = max(y0 + 1, int(0.70 * H))                      # 아래 깊은 glow / 원형 하단 제외
        mx = int(0.12 * W)                                   # 좌우 원형 테두리 제외
        search_region = (y0, y1, mx, W - mx)

    # 검색 영역 (없으면 전체)
    if search_region is None:
        search_region = (0, H, 0, W)
    ys, ye, xs, xe = search_region
    Y, X = np.mgrid[ys:ye, xs:xe]
    bright = gaussian_filter(base, smooth)[ys:ye, xs:xe]      # 밝은 영역(glow 포함)
    spot = sm[ys:ye, xs:xe]                                    # glow 제거한 spot 강조
    # 1) 중앙 C = 밝은 영역(패턴)의 무게중심
    ws = float(bright.sum()) + 1e-9
    Cx = float((X * bright).sum() / ws)
    Cy = float((Y * bright).sum() / ws)
    # 2) C에서 가까울수록 가중 -> 중앙 근처 가장 밝은 spot 선택
    sigma_d = 0.15 * W
    score = spot * np.exp(-((X - Cx) ** 2 + (Y - Cy) ** 2) / (2 * sigma_d ** 2))
    dy, dx = np.unravel_index(np.argmax(score), score.shape)
    y0, x0 = ys + dy, xs + dx

    # window 잘라서 fit
    y1, y2 = max(0, y0 - win), min(H, y0 + win + 1)
    x1, x2 = max(0, x0 - win), min(W, x0 + win + 1)
    patch = g[y1:y2, x1:x2]
    yy, xx = np.mgrid[y1:y2, x1:x2]

    method = 'gaussian2d'
    try:
        p0 = (float(patch.max() - patch.min()), float(x0), float(y0), 3.0, 3.0, float(patch.min()))
        popt, _ = curve_fit(_gauss2d, (xx.ravel(), yy.ravel()), patch.ravel(), p0=p0, maxfev=8000)
        A, xc, yc, sx, sy, off = popt
        sigma = (abs(sx) + abs(sy)) / 2
        if not (x1 <= xc <= x2 and y1 <= yc <= y2 and 0.3 < sigma < win):
            raise ValueError("fit out of bounds")
    except Exception:
        # fit 실패(주로 포화 spot) -> 밝은 영역(>50% 강도)의 무게중심. 그것도 안 되면 거친 argmax.
        thr = patch.min() + 0.5 * (patch.max() - patch.min())
        mask = patch >= thr
        if mask.sum() >= 3:
            xc, yc = float(xx[mask].mean()), float(yy[mask].mean())
            sigma = float(np.sqrt(mask.sum() / np.pi))      # 밝은 영역 면적 -> 등가 반경
            method = 'centroid'
        else:
            xc, yc, sigma, method = float(x0), float(y0), float(smooth * 2), 'coarse'
        A, off = float(patch.max()), float(patch.min())

    return {'x': float(xc), 'y': float(yc), 'sigma': float(sigma), 'method': method,
            'coarse': (int(x0), int(y0)), 'A': float(A), 'offset': float(off)}


def draw_peak(img, res, roi_k=5):
    """원본에 peak 십자(빨강) + ROI 박스(초록) 그려서 RGB uint8 반환 (대비 스트레치)."""
    g = img.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    lo, hi = np.percentile(g, 70), np.percentile(g, 99.8)   # 밝은 특징(spot) 위주 대비 (glow 눌러서 spot 보이게)
    vis = np.clip((g - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    vis = np.stack([vis, vis, vis], axis=-1)
    H, W = g.shape
    xc, yc = int(round(res['x'])), int(round(res['y']))
    r = max(4, int(round(roi_k * res['sigma'])))
    # crosshair (red)
    vis[max(0, yc - 1):yc + 2, :, 0] = 255; vis[max(0, yc - 1):yc + 2, :, 1:] = 0
    vis[:, max(0, xc - 1):xc + 2, 0] = 255; vis[:, max(0, xc - 1):xc + 2, 1:] = 0
    # ROI box (green)
    x1, x2 = max(0, xc - r), min(W - 1, xc + r)
    y1, y2 = max(0, yc - r), min(H - 1, yc + r)
    for (a, b, c, d) in [(y1, y1 + 2, x1, x2), (y2 - 1, y2 + 1, x1, x2), (y1, y2, x1, x1 + 2), (y1, y2, x2 - 1, x2 + 1)]:
        vis[a:b, c:d, 1] = 255; vis[a:b, c:d, 0] = 0; vis[a:b, c:d, 2] = 0
    return vis
