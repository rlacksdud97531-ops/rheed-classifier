import os, re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score

# ----------------------------- CONFIG -----------------------------
DATA_DIR    = os.environ.get('RHEED_DATA', 'balanced_dataset')   # V2 비교: $env:RHEED_DATA='zoom_dataset'
IMG_SIZE    = (224, 224)
BATCH       = 16
SEED        = 42
EPOCHS_HEAD = 15
EPOCHS_FT   = 25            # 미세조정 에포크 확보
PATIENCE    = 6
FT_LAYERS   = 30
DISPLAY     = {'Anomalies': 'Modulated'}   # 폴더명 -> 표시 이름 (Anomalies 폴더 내용 = Modulated)
EXTS        = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

# ---- 상용화 옵션 ----
MERGE_MIXED       = True                                          # Modulated+Anomalous Spots -> 'Mixed' (3-class)
MERGE_MAP         = {'Anomalies': 'Mixed', 'Anomalous Spots': 'Mixed'}
UNCLEAR_THRESHOLD = 0.60                                          # 최대 softmax 확률 < 이 값 -> 'Unclear'(판단 보류, 학습 클래스 아님)

keras.utils.set_random_seed(SEED)


def growth_id(fname):
    m = re.match(r'(\d{6}[A-Za-z]*)', fname)
    return m.group(1) if m else os.path.splitext(fname)[0]


def list_dataset(root):
    raw = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    rename = (lambda c: MERGE_MAP.get(c, c)) if MERGE_MIXED else (lambda c: c)
    classes = sorted(set(rename(c) for c in raw))          # 병합 시 ['Mixed','Spotty','Streaks']
    cidx = {c: i for i, c in enumerate(classes)}
    paths, labels, groups = [], [], []
    for c in raw:
        ci = cidx[rename(c)]                               # 폴더 -> (병합된) 클래스 인덱스
        for dp, _, fs in os.walk(os.path.join(root, c)):
            for f in fs:
                if f.lower().endswith(EXTS):
                    paths.append(os.path.join(dp, f))
                    labels.append(ci)
                    groups.append(growth_id(f))
    return np.array(paths), np.array(labels), np.array(groups), classes


# ------------------- tf.data 파이프라인 (메모리 절약) -------------------
def parse_image(path, label, augment=False):
    img = tf.io.read_file(path)
    # decode_image는 16-bit PNG(mode I;16)를 [0,255]로 올바르게 디코딩 (PIL은 흰색으로 포화시킴)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])                       # 정적 rank 보장 (배치 에러 방지)
    img = tf.image.resize(img, IMG_SIZE, method='bicubic')   # RHEED 패턴 보호용 bicubic

    if augment:
        # RHEED 안전 증강: 거리/위치 왜곡 없는 것만. max_delta는 절대값이므로 255 스케일 기준!
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=38.0)     # ≈ 0.15 * 255
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.clip_by_value(img, 0.0, 255.0)

    return img, label


def make_dataset(paths, labels, batch_size, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(lambda p, l: parse_image(p, l, augment), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


# ------------------- 모델 -------------------
def build_model(n_classes):
    inp = keras.Input(shape=(*IMG_SIZE, 3))
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3), pooling='avg')
    backbone.trainable = False
    x = backbone(inp, training=False)     # BN 추론모드 유지(미세조정 시 통계 파괴 방지)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inp, out), backbone


# ------------------- 'Unclear' 보류 분석 (추론 시 신뢰도 게이트, 학습 클래스 아님) -------------------
def abstain_analysis(oof_probs, y_true, disp, default_thr):
    conf = oof_probs.max(axis=1)
    pred = oof_probs.argmax(axis=1)
    print("\n--- 'Unclear' 보류 분석 (OOF 신뢰도 기준) ---")
    print(" thresh | Unclear비율 | 분류된것 정확도 | 분류된것 macroF1")
    for t in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        keep = conf >= t
        if keep.sum() > 0:
            acc = float((pred[keep] == y_true[keep]).mean())
            mf1 = f1_score(y_true[keep], pred[keep], average='macro', zero_division=0)
        else:
            acc = mf1 = float('nan')
        mark = "  <- default" if abs(t - default_thr) < 1e-9 else ""
        print(f"  {t:.2f}  |   {100*(1-keep.mean()):5.1f}%   |     {acc:.3f}      |    {mf1:.3f}{mark}")

    # 배포 임계값에서 운영자가 실제로 보게 될 4-버킷 분포
    keep = conf >= default_thr
    names = list(disp) + ['Unclear']
    final = np.where(keep, pred, len(disp))
    vals, cnts = np.unique(final, return_counts=True)
    print(f"\n배포 임계값 {default_thr}: 전체 {len(y_true)}장 중 "
          f"{int((~keep).sum())}장({100*(1-keep.mean()):.1f}%)이 'Unclear'로 보류")
    print(" 운영자 출력 분포:", {names[v]: int(c) for v, c in zip(vals, cnts)})


# ------------------- Safety-First 정책: 문제(Mixed)는 민감하게, 정상은 보수적으로 -------------------
def safety_first_decide(probs, classes, clean_conf, mixed_sens):
    """결정 인덱스 반환 (0..n-1 = 클래스, n = Unclear).
       ① P(Mixed) >= mixed_sens -> Mixed(문제, 낮은 문턱)  ② top(정상) >= clean_conf -> 정상  ③ else Unclear"""
    mix = classes.index('Mixed')
    top = probs.argmax(1)
    topp = probs[np.arange(len(probs)), top]
    flag_mix   = probs[:, mix] >= mixed_sens
    conf_clean = (topp >= clean_conf) & (top != mix)
    return np.where(flag_mix, mix, np.where(conf_clean, top, len(classes)))


def safety_first_report(probs, y_true, classes):
    if 'Mixed' not in classes:
        return
    mix = classes.index('Mixed')
    clean = np.array([i for i in range(len(classes)) if i != mix])
    is_problem = (y_true == mix)
    print("\n--- Safety-First 정책 시뮬레이션 (OOF 기준) ---")
    print(" 정상기준 | 문제민감도 | 문제잡기(Mixed재현) | 놓친문제%(위험) | 오경보% | Unclear% | 정상자동정확도")
    for clean_conf in (0.75, 0.80, 0.85):
        for mixed_sens in (0.30, 0.40, 0.50):
            d = safety_first_decide(probs, classes, clean_conf, mixed_sens)
            mixed_recall = np.mean(d[is_problem] == mix) if is_problem.any() else float('nan')
            missed       = np.mean(np.isin(d[is_problem], clean)) if is_problem.any() else float('nan')
            false_alarm  = np.mean(d[~is_problem] == mix) if (~is_problem).any() else float('nan')
            unclear      = np.mean(d == len(classes))
            auto_clean   = np.isin(d, clean)
            clean_acc    = np.mean(d[auto_clean] == y_true[auto_clean]) if auto_clean.any() else float('nan')
            print(f"   {clean_conf:.2f}   |    {mixed_sens:.2f}    |       {mixed_recall:.3f}        |"
                  f"     {100*missed:4.1f}%      | {100*false_alarm:5.1f}% |  {100*unclear:4.1f}%  |     {clean_acc:.3f}")


# ------------------------------- MAIN (5-Fold Group CV + OOF) -------------------------------
def main():
    paths, labels, groups, classes = list_dataset(DATA_DIR)
    n = len(classes)
    disp = [DISPLAY.get(c, c) for c in classes]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_predictions = np.zeros((len(paths), n))
    fold_f1s = []

    print(f"총 이미지 수: {len(paths)} | 클래스 수: {n} | 표시 이름: {disp}")
    print("=== 5-Fold Group CV 시작 (성장 단위 분리, 누수 없음) ===")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(paths, labels, groups)):
        print(f"\n--- FOLD {fold + 1} / 5 ---")
        # 누수 체크: 같은 성장(group)이 train/val에 동시에 있으면 즉시 실패
        assert not (set(groups[train_idx]) & set(groups[val_idx])), f"Fold {fold+1} 데이터 누수 검출!"

        # class weight (역빈도, 0-나눗셈 가드)
        counts = np.bincount(labels[train_idx], minlength=n)
        cw = {i: len(train_idx) / (n * max(counts[i], 1)) for i in range(n)}

        train_ds = make_dataset(paths[train_idx], labels[train_idx], BATCH, augment=True,  shuffle=True)
        val_ds   = make_dataset(paths[val_idx],   labels[val_idx],   BATCH, augment=False, shuffle=False)

        model, backbone = build_model(n)
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=PATIENCE, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=3, min_lr=1e-6),
        ]

        # Stage 1: 헤드 학습 (백본 동결)
        model.compile(optimizer=keras.optimizers.AdamW(1e-3),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
                  class_weight=cw, callbacks=callbacks, verbose=2)

        # Stage 2: 백본 상단 미세조정
        backbone.trainable = True
        for l in backbone.layers[:-FT_LAYERS]:
            l.trainable = False
        model.compile(optimizer=keras.optimizers.AdamW(5e-5),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT,
                  class_weight=cw, callbacks=callbacks, verbose=2)

        # OOF 예측 (val_ds는 shuffle=False라 순서 보존 -> val_idx와 정렬됨)
        val_preds = model.predict(val_ds, verbose=0)
        oof_predictions[val_idx] = val_preds
        fold_f1 = f1_score(labels[val_idx], val_preds.argmax(axis=1), average='macro')
        fold_f1s.append(fold_f1)
        print(f"Fold {fold + 1} Macro-F1: {fold_f1:.4f}")

    # ------------------------- 최종 OOF 평가 -------------------------
    print("\n================ 최종 검증 리포트 (Out-of-Fold, 전체 데이터) ================")
    oof_pred_labels = oof_predictions.argmax(axis=1)
    print(classification_report(labels, oof_pred_labels, target_names=disp, digits=3))
    print(f"5-Fold 평균 Macro-F1 : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"전체 OOF Macro-F1    : {f1_score(labels, oof_pred_labels, average='macro'):.4f}")
    print(f"전체 OOF Balanced-Acc: {balanced_accuracy_score(labels, oof_pred_labels):.4f}")

    # 상용화: 'Unclear' 보류 임계값별 trade-off + 배포값에서의 운영자 출력
    abstain_analysis(oof_predictions, labels, disp, UNCLEAR_THRESHOLD)

    # OOF 저장(추후 정책 실험은 재학습 없이 이 파일로) + Safety-First 정책 시뮬레이션
    os.makedirs('results', exist_ok=True)
    np.savez('results/oof.npz', probs=oof_predictions, labels=labels, classes=np.array(classes))
    print("\nOOF 확률 저장 -> results/oof.npz (재학습 없이 정책 실험 가능)")
    safety_first_report(oof_predictions, labels, classes)


if __name__ == '__main__':
    main()
