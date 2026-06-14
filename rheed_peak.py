"""
rheed_peak.py — RHEED 중심(specular, (00)) streak 자동 검출.

핵심 아이디어(대칭성 이용): 회절 패턴은 (00) 중심 streak 기준 좌우 대칭이다.
  1) shadow edge 바로 아래 띠에서 가로 프로파일 -> streak 들의 x 위치를 모두 찾는다.
  2) 개수가 짝수면 가장 약한 streak 하나를 빼서 홀수로 만든다.
  3) "가운데(중앙값 위치) streak" = 중심 (00). (밝기에 의존 안 하므로 glow에 안 속음)
  4) 그 streak을 따라 세로 프로파일(글로우 제거) -> specular spot 높이(y).
  5) (x,y) 주변 2D Gaussian fit으로 sub-pixel 중심 + 폭(sigma). 실패 시 centroid/coarse fallback.

반환 dict에 streaks_x(검출된 streak x들), spacing(중앙 streak 간격, px)도 담아
다음 단계(streak 간격 -> lattice constant) 분석에 바로 쓸 수 있게 한다.
"""
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter, white_tophat
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def load_gray(path):
    """16-bit 안전 로드 -> 2D float32 그레이스케일."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=1, expand_animations=False)
    return img.numpy().squeeze().astype(np.float32)


def _gauss2d(coords, A, xc, yc, sx, sy, off):
    x, y = coords
    return (A * np.exp(-((x - xc) ** 2 / (2 * sx ** 2) + (y - yc) ** 2 / (2 * sy ** 2))) + off).ravel()


def _shadow_edge(g):
    """shadow edge(밝은 패턴이 시작되는 위쪽 경계) 행. 실패 시 0.15H."""
    H = g.shape[0]
    try:
        import rheed_crop
        edge = int(rheed_crop.find_crop_row(g))
    except Exception:
        edge = int(0.15 * H)
    return max(0, min(edge, int(0.5 * H)))


def find_center_peak(img, smooth=3.0, win=20, search_region=None, skip_top=True):
    """중심 (00) streak 위의 specular 중심 + 폭을 대칭성으로 검출.
       img: 2D 또는 (H,W,3).
       반환 dict: x, y, sigma, method, coarse, A, offset, streaks_x, spacing, n_streaks."""
    g = img.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    H, W = g.shape
    base = median_filter(g, size=3)                                    # hot pixel 제거
    th = white_tophat(base, size=max(15, int(0.02 * max(H, W))))       # 넓은 glow 제거 -> compact streak만
    edge = _shadow_edge(g)

    # 1) shadow edge 아래 띠에서 가로 프로파일 -> streak x 위치
    y0, y1 = edge, min(H, edge + int(0.22 * H))
    prof = gaussian_filter1d(th[y0:y1].sum(axis=0).astype(np.float64), 5)
    mx = int(0.10 * W)                                                 # 좌우 원형 테두리 제외
    prof[:mx] = 0.0
    prof[-mx:] = 0.0
    pmax = float(prof.max())
    if pmax > 0:
        peaks, props = find_peaks(prof, prominence=pmax * 0.10, distance=max(1, int(0.045 * W)))
    else:
        peaks, props = np.array([], dtype=int), {'prominences': np.array([])}

    # 2) 짝수면 가장 약한 streak 제거 -> 홀수
    peaks = np.sort(peaks)
    if len(peaks) % 2 == 0 and len(peaks) > 0:
        proms = props['prominences'][np.argsort(peaks)] if len(props['prominences']) == len(peaks) else None
        drop = int(np.argmin(proms)) if proms is not None else 0
        peaks = np.delete(peaks, drop)
    n_streaks = int(len(peaks))

    # 3) 가운데(중앙값 위치) streak = 중심 (00)
    cx = int(peaks[len(peaks) // 2]) if n_streaks else W // 2

    # 4) 중앙 streak 따라 세로 프로파일(글로우 제거) -> specular 높이 y
    vh = min(H, edge + int(0.45 * H))
    band = th[edge:vh, max(0, cx - 12):min(W, cx + 13)]
    if band.size:
        vcol = gaussian_filter1d(band.sum(axis=1).astype(np.float64), 3)
        cy = edge + int(np.argmax(vcol))
    else:
        cy = (y0 + y1) // 2

    spacing = float(np.median(np.diff(peaks))) if n_streaks >= 2 else float('nan')

    # 5) (cx,cy) 주변 2D Gaussian fit -> sub-pixel 중심 + sigma (실패 시 centroid/coarse)
    yy0, yy1 = max(0, cy - win), min(H, cy + win + 1)
    xx0, xx1 = max(0, cx - win), min(W, cx + win + 1)
    patch = base[yy0:yy1, xx0:xx1]
    yy, xx = np.mgrid[yy0:yy1, xx0:xx1]
    method = 'gaussian2d'
    try:
        p0 = (float(patch.max() - patch.min()), float(cx), float(cy), 3.0, 3.0, float(patch.min()))
        popt, _ = curve_fit(_gauss2d, (xx.ravel(), yy.ravel()), patch.ravel(), p0=p0, maxfev=8000)
        A, xc, yc, sx, sy, off = popt
        sigma = (abs(sx) + abs(sy)) / 2
        if not (xx0 <= xc <= xx1 and yy0 <= yc <= yy1 and 0.3 < sigma < win):
            raise ValueError("fit out of bounds")
    except Exception:
        thr = patch.min() + 0.5 * (patch.max() - patch.min())
        mask = patch >= thr
        if mask.sum() >= 3:
            xc, yc = float(xx[mask].mean()), float(yy[mask].mean())
            sigma = float(np.sqrt(mask.sum() / np.pi))
            method = 'centroid'
        else:
            xc, yc, sigma, method = float(cx), float(cy), float(smooth * 2), 'coarse'
        A, off = float(patch.max()), float(patch.min())

    return {'x': float(xc), 'y': float(yc), 'sigma': float(sigma), 'method': method,
            'coarse': (int(cx), int(cy)), 'A': float(A), 'offset': float(off),
            'streaks_x': [int(p) for p in peaks], 'spacing': spacing, 'n_streaks': n_streaks}


def draw_peak(img, res, roi_k=5, show_streaks=False):
    """원본에 peak 십자(빨강) + ROI 박스(초록) 그려서 RGB uint8 반환 (대비 스트레치).
       show_streaks=True면 검출된 streak 위치를 옅은 초록 세로선으로 표시."""
    g = img.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    lo, hi = np.percentile(g, 70), np.percentile(g, 99.8)   # 밝은 특징(spot) 위주 대비 (glow 눌러서 spot 보이게)
    vis = np.clip((g - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    vis = np.stack([vis, vis, vis], axis=-1)
    H, W = g.shape
    xc, yc = int(round(res['x'])), int(round(res['y']))
    r = max(4, int(round(roi_k * res['sigma'])))
    if show_streaks:
        for px in res.get('streaks_x', []):
            if px == xc:
                continue
            vis[:, max(0, px):min(W, px + 1), 1] = 200
            vis[:, max(0, px):min(W, px + 1), 0] = 0
            vis[:, max(0, px):min(W, px + 1), 2] = 0
    # crosshair (red)
    vis[max(0, yc - 1):yc + 2, :, 0] = 255; vis[max(0, yc - 1):yc + 2, :, 1:] = 0
    vis[:, max(0, xc - 1):xc + 2, 0] = 255; vis[:, max(0, xc - 1):xc + 2, 1:] = 0
    # ROI box (green)
    x1, x2 = max(0, xc - r), min(W - 1, xc + r)
    y1, y2 = max(0, yc - r), min(H - 1, yc + r)
    for (a, b, c, d) in [(y1, y1 + 2, x1, x2), (y2 - 1, y2 + 1, x1, x2), (y1, y2, x1, x1 + 2), (y1, y2, x2 - 1, x2 + 1)]:
        vis[a:b, c:d, 1] = 255; vis[a:b, c:d, 0] = 0; vis[a:b, c:d, 2] = 0
    return vis
