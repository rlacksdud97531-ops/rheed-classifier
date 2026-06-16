"""
rheed_crop.py — RHEED 원형 이미지에서 shadow edge 위(어두운 영역)를 자동으로 잘라냄.
원리: 중앙 세로 스트립의 행별 밝기에서 '가장 길게 지속되는 밝은 띠(=패턴+glow)'의 시작을 edge로.
      위쪽의 고립된 밝은 노이즈(반사 등)는 짧은 밝은 구간이라 자동으로 무시됨.
사용:  python rheed_crop.py <폴더>  -> crop_preview/ 에 (크롭선 그린 원본 + 잘린 결과) 저장
"""
import os, glob, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from PIL import Image


def _load(path):
    return tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False).numpy()


def find_crop_row(img, central=(0.30, 0.70), smooth=11, dark_frac=0.15, min_band=0.20, margin_run=0.012):
    """'가장 길게 연속으로 밝은 띠(=패턴+glow)'를 찾아, 그 시작 위를 shadow edge로 잡는다.
       당신 아이디어(연속으로 밝은 패턴 영역 찾기)의 robust 버전:
         - 위쪽 고립 노이즈 = 짧은 밝은 구간 -> 탈락
         - 바닥 어두운 테두리 / 위가 밝은 변형 -> 영향 없음 (가장 큰 밝은 띠만 선택)
       보완: (1) '검정'=적응형 임계값 이하  (2) 가운데 '띠' median  (3) 작은 어두운 틈은 무시."""
    a = np.asarray(img).astype(np.float32)
    g = a[..., 1] if a.ndim == 3 else a                  # green 채널 (RHEED 형광)
    H, W = g.shape
    c0, c1 = int(central[0] * W), int(central[1] * W)
    row = np.median(g[:, c0:c1], axis=1)                 # 가운데 띠 행별 median
    k = max(1, int(smooth))
    if k > 1:
        row = np.convolve(row, np.ones(k) / k, mode='same')
    base, peak = np.percentile(row, 20), np.percentile(row, 95)
    if peak - base < 1e-6:
        return 0
    bright = row > base + dark_frac * (peak - base)      # 이 값 초과 = 밝음
    gap = max(3, int(margin_run * H))                    # 이보다 짧은 어두운 틈은 같은 띠로 (스팟 사이 빈틈 무시)
    # 가장 긴 연속 'bright' 띠 찾기 (짧은 어두운 틈은 메움)
    best_s, best_e, s, dark = -1, -1, None, 0
    for y in range(H):
        if bright[y]:
            if s is None:
                s = y
            dark = 0
        else:
            if s is not None:
                dark += 1
                if dark > gap:                           # 띠 종료
                    if y - dark - s > best_e - best_s:
                        best_s, best_e = s, y - dark
                    s, dark = None, 0
    if s is not None and H - s > best_e - best_s:
        best_s, best_e = s, H
    if best_s < 0 or (best_e - best_s) < min_band * H:   # 충분히 큰 밝은 띠가 없으면 안 자름
        return 0
    return best_s


def auto_crop(img, margin=0.02, **kw):
    a = np.asarray(img)
    top = max(0, int(find_crop_row(a, **kw) - margin * a.shape[0]))
    return a[top:]


def pattern_box(img, side_frac=0.12, **kw):
    """패턴 중심 직사각 박스 좌표 -> (top, bottom, left, right).
       top = shadow edge(find_crop_row), 좌우·아래 = 밝은 내용(원 내부) 경계.
       balanced_dataset처럼 '원형 테두리 없는 패턴 직사각형'을 만들기 위함."""
    a = np.asarray(img).astype(np.float32)
    g = a[..., 1] if a.ndim == 3 else a
    H, W = g.shape
    top = find_crop_row(a, **kw)
    reg = g[top:]
    if reg.shape[0] < 0.1 * H:
        return 0, H, 0, W
    base, peak = np.percentile(g, 30), np.percentile(g, 98)
    thr = base + side_frac * (peak - base)
    xs = np.where(np.median(reg, axis=0) > thr)[0]          # 밝은 열 (좌우 경계)
    left, right = (int(xs.min()), int(xs.max()) + 1) if len(xs) else (0, W)
    ys = np.where(np.median(reg[:, left:right], axis=1) > thr)[0]   # 밝은 행 (아래 경계)
    bottom = top + (int(ys.max()) + 1 if len(ys) else reg.shape[0])
    return int(top), int(bottom), int(left), int(right)


def crop_pattern(img, **kw):
    t, b, l, r = pattern_box(img, **kw)
    return np.asarray(img)[t:b, l:r]


def zoom_box(img, band_h=0.60, width_frac=0.72, center=False, **kw):
    """V2 줌 영역 좌표 -> (top, bottom, left, right). 패턴 박스 안 '상단-중앙 밴드'(image2처럼).
       수직 = shadow edge 직하부터 band_h 비율, 수평 = 밝은 열(줄무늬)의 무게중심 ± width_frac/2.
       center=True면 무게중심 대신 패턴 박스의 가로 정중앙을 쓴다."""
    a = np.asarray(img)
    t, b, l, r = pattern_box(a, **kw)
    g = (a[..., 1] if a.ndim == 3 else a)[t:b, l:r].astype(np.float32)
    ph, pw = g.shape
    if ph < 5 or pw < 5:
        return t, b, l, r
    y1 = t + max(1, int(band_h * ph))                       # 수직 밴드 (상단부)
    if center:
        cx = pw // 2                                        # 가로 정중앙
    else:
        col = g[:max(1, int(0.6 * ph))].mean(axis=0)        # 상단부 열별 밝기 = 줄무늬 위치
        w = np.clip(col - np.percentile(col, 20), 0, None)
        cx = int((np.arange(pw) * w).sum() / w.sum()) if w.sum() > 0 else pw // 2  # 밝기 무게중심
    half = int(width_frac * pw / 2)
    x0, x1 = max(0, cx - half), min(pw, cx + half)
    return t, y1, l + x0, l + x1


def tight_zoom(img, **kw):
    t, b, l, r = zoom_box(img, **kw)
    return np.asarray(img)[t:b, l:r]


def _viz(folder, out='crop_preview', n=8):
    os.makedirs(out, exist_ok=True)
    files = (sorted(glob.glob(os.path.join(folder, '*.png'))) +
             sorted(glob.glob(os.path.join(folder, '*.jpg'))))[:n]
    for f in files:
        img = _load(f); H, W = img.shape[:2]
        t, b, l, r = pattern_box(img)
        vis = img.copy()
        for (y0, y1, x0, x1) in [(t, t + 3, l, r), (b - 3, b, l, r), (t, b, l, l + 3), (t, b, r - 3, r)]:
            vis[max(0, y0):max(0, y1), max(0, x0):max(0, x1), 0] = 255
            vis[max(0, y0):max(0, y1), max(0, x0):max(0, x1), 1] = 0
            vis[max(0, y0):max(0, y1), max(0, x0):max(0, x1), 2] = 0
        Image.fromarray(vis.astype('uint8')).save(os.path.join(out, 'box_' + os.path.basename(f)))
        Image.fromarray(crop_pattern(img).astype('uint8')).save(os.path.join(out, 'crop_' + os.path.basename(f)))
        print(f"  {os.path.basename(f)[:30]:32s} box t,b,l,r=({t},{b},{l},{r}) of {H}x{W}", flush=True)
    print(f"-> {out}/ 에 저장 (box_=박스 표시, crop_=잘린 결과)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--n', type=int, default=8)
    a = ap.parse_args()
    _viz(a.folder, n=a.n)
