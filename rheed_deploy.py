"""
rheed_deploy.py — RHEED 3-class 실시간 분류 배포 모듈

사용:
  python rheed_deploy.py --train     # 전체 데이터로 최종 모델 1개 학습 -> results/rheed_final.keras
  python rheed_deploy.py --demo      # 저장된 모델로 샘플 몇 장 예측 (출력 형식 확인)

실시간 루프에서:
  from rheed_deploy import load_classifier, predict_frame, TemporalSmoother
  model = load_classifier()
  smoother = TemporalSmoother(window=5)
  res = predict_frame(model, frame)          # frame: PNG 경로 또는 numpy 이미지
  stable_label = smoother.update(res['label'])

Safety-First 정책 (OOF로 측정해 확정한 운영점):
  ① P(Mixed) >= MIXED_SENS         -> 'Mixed'   (문제는 민감하게)
  ② top(정상) >= CLEAN_CONF        -> 정상 클래스 (정상은 보수적으로)
  ③ 그 외                          -> 'Unclear' (사람 확인)
"""
import os, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from collections import deque, Counter
import numpy as np
import tensorflow as tf
import keras

# ----------------------------- 배포 설정 -----------------------------
IMG_SIZE    = (224, 224)
CLASSES     = ['Mixed', 'Spotty', 'Streaks']     # 학습 시 폴더 알파벳순(병합)과 동일해야 함
MIX_IDX     = CLASSES.index('Mixed')
MODEL_PATH  = 'results/rheed_final.keras'
CLEAN_CONF  = 0.90      # 정상으로 자동 판정할 최소 확신
MIXED_SENS  = 0.25      # 문제(Mixed)로 플래그할 최소 P(Mixed)

# 운영자/장비용 액션 코드
ACTIONS = {
    'Streaks': (0, 'OK — 2D layer-by-layer (FM)'),
    'Spotty':  (1, 'OK — quantum dots / 3D islands (VW)'),
    'Mixed':   (2, 'ALERT — transitional/rough growth, check'),
    'Unclear': (9, 'REVIEW — low confidence, inspect frame'),
}


# ----------------------------- 입력 전처리 (학습과 동일) -----------------------------
def _to_input(img):
    """PNG 경로(str) 또는 numpy 이미지 -> (1,224,224,3) float32 [0,255]. 16-bit 안전."""
    if isinstance(img, (str, bytes, os.PathLike)):
        x = tf.io.decode_image(tf.io.read_file(str(img)), channels=3, expand_animations=False)
        x = tf.cast(x, tf.float32)
    else:
        arr = np.asarray(img)
        arr = (arr.astype(np.float32) / 256.0) if arr.dtype == np.uint16 else arr.astype(np.float32)
        x = tf.convert_to_tensor(arr)
        if x.ndim == 2:
            x = tf.stack([x, x, x], axis=-1)
        elif x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
    x = tf.image.resize(x, IMG_SIZE)
    return tf.expand_dims(x, 0)


# ----------------------------- Safety-First 결정 -----------------------------
def decide(probs):
    """probs: (n_classes,) softmax -> (label, confidence)."""
    p = np.asarray(probs)
    if p[MIX_IDX] >= MIXED_SENS:                          # ① 문제 의심
        return 'Mixed', float(p[MIX_IDX])
    top = int(p.argmax())
    if top != MIX_IDX and p[top] >= CLEAN_CONF:           # ② 확실한 정상
        return CLASSES[top], float(p[top])
    return 'Unclear', float(p.max())                      # ③ 보류


def predict_frame(model, img):
    """이미지 1장 -> 결과 dict(label, confidence, code, message, probs)."""
    p = model.predict(_to_input(img), verbose=0)[0]
    label, conf = decide(p)
    code, msg = ACTIONS[label]
    return {
        'label': label,
        'confidence': round(conf, 4),
        'code': code,
        'message': msg,
        'probs': {c: round(float(p[i]), 4) for i, c in enumerate(CLASSES)},
    }


# ----------------------------- 시간적 스무딩 (실시간 스트림) -----------------------------
class TemporalSmoother:
    """최근 window 프레임의 다수결로 라벨 안정화. (안전을 더 원하면 Mixed에 가중치 추가 가능)"""
    def __init__(self, window=5):
        self.buf = deque(maxlen=window)

    def update(self, label):
        self.buf.append(label)
        return Counter(self.buf).most_common(1)[0][0]

    def reset(self):
        self.buf.clear()


def load_classifier(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 없음. 먼저 `python rheed_deploy.py --train` 실행하세요.")
    return keras.models.load_model(path)


# ----------------------------- 최종 모델 학습 (전체 데이터, 1회) -----------------------------
def train_final_model():
    """CV는 성능 추정용. 배포 모델은 전체 데이터로 학습한다(조기종료용 소규모 holdout만 분리)."""
    import rheed_cv as R
    from sklearn.model_selection import StratifiedGroupKFold
    assert R.MERGE_MIXED, "rheed_cv.MERGE_MIXED=True 여야 3-class(Mixed)로 학습됩니다."

    paths, labels, groups, classes = R.list_dataset(R.DATA_DIR)
    assert classes == CLASSES, f"클래스 순서 불일치: {classes} != {CLASSES}"
    # 조기종료용 소규모 group-holdout (~14%), 나머지로 학습
    tr, va = next(iter(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=R.SEED)
                       .split(paths, labels, groups)))
    print(f"최종 모델 학습: train {len(tr)} / earlystop-val {len(va)}")

    train_ds = R.make_dataset(paths[tr], labels[tr], R.BATCH, augment=True,  shuffle=True)
    val_ds   = R.make_dataset(paths[va], labels[va], R.BATCH, augment=False, shuffle=False)
    counts = np.bincount(labels[tr], minlength=len(classes))
    cw = {i: len(tr) / (len(classes) * max(counts[i], 1)) for i in range(len(classes))}

    model, backbone = R.build_model(len(classes))
    cbs = [keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=R.PATIENCE, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=3, min_lr=1e-6)]

    model.compile(optimizer=keras.optimizers.AdamW(1e-3),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=R.EPOCHS_HEAD, class_weight=cw, callbacks=cbs, verbose=2)

    backbone.trainable = True
    for l in backbone.layers[:-R.FT_LAYERS]:
        l.trainable = False
    model.compile(optimizer=keras.optimizers.AdamW(5e-5),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=R.EPOCHS_FT, class_weight=cw, callbacks=cbs, verbose=2)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n저장: {MODEL_PATH} | classes={classes} | 정책 CLEAN_CONF={CLEAN_CONF}, MIXED_SENS={MIXED_SENS}")


def _demo():
    import glob
    model = load_classifier()
    sm = TemporalSmoother(window=5)
    sample = []
    for c in ['Streaks', 'Spotty', 'Anomalies']:                # 폴더 기준 샘플
        sample += glob.glob(os.path.join('balanced_dataset', c, '*.png'))[:2]
    for p in sample:
        r = predict_frame(model, p)
        print(f"{os.path.basename(p)[:32]:34s} -> {r['label']:8s} ({r['confidence']:.2f}) "
              f"code={r['code']} | {r['message']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true', help='전체 데이터로 최종 모델 학습')
    ap.add_argument('--demo',  action='store_true', help='저장된 모델로 샘플 예측')
    a = ap.parse_args()
    if a.train:
        train_final_model()
    elif a.demo:
        _demo()
    else:
        print("사용법: python rheed_deploy.py --train   (또는 --demo)")
