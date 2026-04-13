"""
model_utils.py
──────────────
Handles:
  • Loading / caching Keras .h5 models
  • CLAHE preprocessing matching the training pipeline
  • Ensemble inference (probability averaging)
  • Fallback: Hugging Face Inference API  →  Gemini Vision API
  • Retraining a new model in a background thread
"""

import os
import io
import base64
import json
import threading
import logging
import time
import traceback
from typing import Optional, List

import cv2
import numpy as np
import requests
from PIL import Image

log = logging.getLogger(__name__)

# ── TensorFlow — lazy, optional import ────────────────────────────────────────
# TF is NOT required at startup. It is loaded on the first prediction request.
# If TF is unavailable (wrong Python version, unsupported platform, etc.)
# the app falls back to Hugging Face / Gemini Vision APIs automatically.

_tf = None
_tf_unavailable = False          # set True once we know TF can't load
_tf_unavailable_reason = ""

def _get_tf():
    global _tf, _tf_unavailable, _tf_unavailable_reason
    if _tf_unavailable:
        return None
    if _tf is None:
        try:
            import tensorflow as tf
            _tf = tf
        except ImportError as exc:
            _tf_unavailable = True
            _tf_unavailable_reason = (
                f"TensorFlow not installed ({exc}). "
                "Run  python install.py  to install it, or configure a "
                "Hugging Face / Gemini API key in Admin → Settings for fallback mode."
            )
            log.warning("TensorFlow unavailable: %s", _tf_unavailable_reason)
        except Exception as exc:
            _tf_unavailable = True
            _tf_unavailable_reason = f"TensorFlow failed to load: {exc}"
            log.error("TensorFlow load error: %s", exc)
    return _tf


def tf_status() -> dict:
    """Return TF availability info for health checks and admin display."""
    if _tf_unavailable:
        return {"available": False, "reason": _tf_unavailable_reason, "version": None}
    if _tf is not None:
        return {"available": True, "reason": None, "version": _tf.__version__}
    return {"available": None, "reason": "Not yet loaded", "version": None}

# ── model cache ────────────────────────────────────────────────────────────────
_loaded_models: dict = {}          # key → keras model
_model_lock = threading.Lock()


def _make_compat_objects():
    """
    Return custom_objects dict that strips unknown Keras-3 config keys
    (e.g. quantization_config, lora_rank) that old .h5 files may carry
    but current TF/Keras build doesn't accept.
    """
    tf = _get_tf()
    if tf is None:
        return {}

    _STRIP_KEYS = ('quantization_config', 'lora_rank', 'lora_alpha',
                   'dtype_policy', 'use_legacy_activation')

    class _CompatLayer:
        """Mixin: strip unknown config keys before calling the real from_config."""
        @classmethod
        def from_config(cls, config):
            for k in _STRIP_KEYS:
                config.pop(k, None)
            return super().from_config(config)

    class CompatDense(_CompatLayer, tf.keras.layers.Dense):
        pass

    class CompatBatchNorm(_CompatLayer, tf.keras.layers.BatchNormalization):
        pass

    class CompatDropout(_CompatLayer, tf.keras.layers.Dropout):
        pass

    class CompatConv2D(_CompatLayer, tf.keras.layers.Conv2D):
        pass

    class CompatDepthwiseConv2D(_CompatLayer, tf.keras.layers.DepthwiseConv2D):
        pass

    return {
        'Dense':                CompatDense,
        'BatchNormalization':   CompatBatchNorm,
        'Dropout':              CompatDropout,
        'Conv2D':               CompatConv2D,
        'DepthwiseConv2D':      CompatDepthwiseConv2D,
    }


def load_model(model_key: str, model_path: str):
    """Load (or return cached) a Keras model with TF 2.20 / Keras 3 compat shims."""
    with _model_lock:
        if model_key not in _loaded_models:
            if not os.path.exists(model_path):
                log.warning("Model file not found: %s", model_path)
                return None
            try:
                tf = _get_tf()
                if tf is None:
                    log.warning(
                        "TensorFlow not available — cannot load model %s. "
                        "Configure a Hugging Face or Gemini API key in Admin → Settings "
                        "for fallback inference.",
                        model_key,
                    )
                    return None
                log.info("Loading model %s from %s …", model_key, model_path)
                custom_objects = _make_compat_objects()
                _loaded_models[model_key] = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects=custom_objects,
                )
                log.info("Model %s loaded successfully.", model_key)
            except MemoryError:
                log.error(
                    "Out of memory loading model %s — server RAM too low. "
                    "Disable some models in Admin → Models to reduce memory use.",
                    model_key,
                )
                return None
            except Exception as exc:
                log.error("Failed to load %s: %s", model_key, exc)
                return None
    return _loaded_models.get(model_key)


def unload_model(model_key: str):
    with _model_lock:
        _loaded_models.pop(model_key, None)


# ── image preprocessing ────────────────────────────────────────────────────────

def clahe_preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE on the L channel (LAB colour space) – matches training pipeline.
    Returns float32 array normalised to [0, 1], shape (224, 224, 3).
    """
    img_uint8 = (img_rgb * 255).astype(np.uint8) if img_rgb.max() <= 1.0 else img_rgb.astype(np.uint8)
    lab  = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb  = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb.astype(np.float32) / 255.0


def prepare_image(file_bytes: bytes, image_size=(224, 224)) -> np.ndarray:
    """
    Decode image bytes → CLAHE-preprocessed numpy array ready for model input.
    Returns shape (1, H, W, 3).
    """
    pil_img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    pil_img = pil_img.resize(image_size, Image.LANCZOS)
    arr     = np.array(pil_img, dtype=np.float32)        # (H, W, 3) uint8→float
    arr     = clahe_preprocess(arr)                       # CLAHE + normalise
    return np.expand_dims(arr, axis=0)                    # (1, H, W, 3)


# ── local ensemble inference ───────────────────────────────────────────────────

def predict_ensemble(
    file_bytes:   bytes,
    models_config: dict,
    class_labels: list,
    image_size=(224, 224)
) -> dict:
    """
    Run probability-averaging ensemble over all enabled, loaded models.

    Returns dict:
        prediction  str
        confidence  float
        probs       list[{label, prob}]
        models_used list[str]
        error       str | None
    """
    img_batch = prepare_image(file_bytes, image_size)
    all_probs = []  # type: List[np.ndarray]
    models_used = []

    for key, info in models_config.items():
        if not info.get('enabled', True):
            continue
        model_path = os.path.join(info['model_dir'], info['filename'])
        model = load_model(key, model_path)
        if model is None:
            continue
        try:
            preds = model.predict(img_batch, verbose=0)[0]   # shape (4,)
            all_probs.append(preds)
            models_used.append(info['display_name'])
        except Exception as exc:
            log.error("Prediction failed for %s: %s", key, exc)

    if not all_probs:
        return {'error': 'No models available for inference.'}

    avg_probs = np.mean(all_probs, axis=0)
    idx = int(np.argmax(avg_probs))

    return {
        'prediction': class_labels[idx],
        'confidence': float(avg_probs[idx]),
        'probs':      [{'label': l, 'prob': float(p)}
                       for l, p in zip(class_labels, avg_probs)],
        'models_used': models_used,
        'error':       None,
    }


# ── Hugging Face fallback ──────────────────────────────────────────────────────

# Map HF labels → our standard labels where possible
_HF_LABEL_MAP = {
    'cataract':             'Cataract',
    'diabetic_retinopathy': 'Diabetic Retinopathy',
    'diabetic retinopathy': 'Diabetic Retinopathy',
    'glaucoma':             'Glaucoma',
    'normal':               'Normal',
}


def _call_hf_api(file_bytes: bytes, api_url: str, api_key: str,
                 class_labels: list) -> Optional[dict]:
    """
    Call Hugging Face Inference API (image classification).
    Returns dict like predict_ensemble, or None on failure.
    """
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    try:
        resp = requests.post(
            api_url, headers=headers, data=file_bytes, timeout=30
        )
        resp.raise_for_status()
        raw = resp.json()
        if isinstance(raw, list) and raw:
            # Build probability vector aligned to our class_labels
            probs = [0.0] * len(class_labels)
            for item in raw:
                lbl = _HF_LABEL_MAP.get(item.get('label', '').lower())
                if lbl and lbl in class_labels:
                    probs[class_labels.index(lbl)] = item['score']
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            idx = int(np.argmax(probs))
            return {
                'prediction': class_labels[idx],
                'confidence': float(probs[idx]),
                'probs':      [{'label': l, 'prob': float(p)}
                               for l, p in zip(class_labels, probs)],
                'models_used': ['HuggingFace API'],
                'error':       None,
            }
    except Exception as exc:
        log.warning("HF API call failed: %s", exc)
    return None


# ── Gemini Vision fallback ─────────────────────────────────────────────────────

_GEMINI_PROMPT = """You are a clinical AI assistant specialised in fundus image analysis.
Analyse this fundus (retinal) photograph and classify it into exactly one of these four categories:
  1. Cataract
  2. Diabetic Retinopathy
  3. Glaucoma
  4. Normal

Respond with ONLY a JSON object like:
{"prediction": "Glaucoma", "confidence": 0.87, "reasoning": "...brief clinical reasoning..."}

Do NOT include any other text outside the JSON object."""


def _call_gemini_api(file_bytes: bytes, api_key: str,
                     api_url: str, class_labels: list) -> Optional[dict]:
    """
    Call Google Gemini Vision API for image classification.
    Returns dict like predict_ensemble, or None on failure.
    """
    if not api_key:
        return None
    try:
        b64 = base64.b64encode(file_bytes).decode()
        payload = {
            "contents": [{
                "parts": [
                    {"text": _GEMINI_PROMPT},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
                ]
            }],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 300}
        }
        resp = requests.post(
            f"{api_url}?key={api_key}",
            json=payload, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data.get('candidates', [{}])[0]
                    .get('content', {})
                    .get('parts', [{}])[0]
                    .get('text', ''))
        parsed = json.loads(text)
        pred   = parsed.get('prediction', '')
        conf   = float(parsed.get('confidence', 0.5))
        # Build uniform prob vector
        probs  = [0.0] * len(class_labels)
        if pred in class_labels:
            probs[class_labels.index(pred)] = conf
            remaining = (1.0 - conf) / max(len(class_labels) - 1, 1)
            for i, lbl in enumerate(class_labels):
                if lbl != pred:
                    probs[i] = remaining
        return {
            'prediction':  pred if pred in class_labels else class_labels[0],
            'confidence':  conf,
            'probs':       [{'label': l, 'prob': p}
                            for l, p in zip(class_labels, probs)],
            'models_used': ['Gemini Vision API'],
            'reasoning':   parsed.get('reasoning', ''),
            'error':       None,
        }
    except Exception as exc:
        log.warning("Gemini API call failed: %s", exc)
    return None


# ── Master predict function ────────────────────────────────────────────────────

def predict(
    file_bytes:         bytes,
    models_config:      dict,
    class_labels:       list,
    confidence_threshold: float,
    fallback_enabled:   bool,
    fallback_priority:  str,      # 'huggingface' | 'gemini'
    hf_api_url:         str,
    hf_api_key:         str,
    gemini_api_key:     str,
    gemini_api_url:     str,
    image_size=(224, 224),
) -> dict:
    """
    Full prediction pipeline with automatic fallback.

    Returns result dict with extra keys:
        method          str   ('ensemble' | 'fallback_hf' | 'fallback_gemini' | 'fallback_combined')
        fallback_used   bool
        fallback_note   str
    """
    result = predict_ensemble(file_bytes, models_config, class_labels, image_size)

    if result.get('error'):
        # Local models totally failed → must use fallback
        result['fallback_used']  = True
        result['fallback_note']  = 'Local models unavailable. Using backup API.'
        result['method']         = 'fallback_only'
    else:
        result['fallback_used'] = False
        result['fallback_note'] = ''
        result['method']        = 'ensemble'

    # ── Decide whether to invoke fallback ──────────────────────────────────────
    low_confidence = (not result.get('error') and
                      result.get('confidence', 1.0) < confidence_threshold)

    if fallback_enabled and (result.get('error') or low_confidence):
        fb_result = None
        method_tag = ''

        apis = (['huggingface', 'gemini'] if fallback_priority == 'huggingface'
                else ['gemini', 'huggingface'])

        for api_name in apis:
            if api_name == 'huggingface':
                fb_result = _call_hf_api(file_bytes, hf_api_url, hf_api_key, class_labels)
                if fb_result:
                    method_tag = 'fallback_hf'
                    break
            elif api_name == 'gemini':
                fb_result = _call_gemini_api(file_bytes, gemini_api_key,
                                             gemini_api_url, class_labels)
                if fb_result:
                    method_tag = 'fallback_gemini'
                    break

        if fb_result:
            if result.get('error'):
                # Local failed completely – use API result as primary
                result = fb_result
                result['method']       = method_tag
                result['fallback_used']= True
                result['fallback_note']= f'Local models unavailable. Result from {method_tag}.'
            else:
                # Low confidence – blend & re-validate
                ensemble_probs  = np.array([p['prob'] for p in result['probs']])
                fallback_probs  = np.array([p['prob'] for p in fb_result['probs']])
                # Weighted average: ensemble 60%, fallback 40%
                combined = 0.6 * ensemble_probs + 0.4 * fallback_probs
                idx      = int(np.argmax(combined))
                result['probs']       = [{'label': l, 'prob': float(p)}
                                         for l, p in zip(class_labels, combined)]
                result['prediction']  = class_labels[idx]
                result['confidence']  = float(combined[idx])
                result['method']      = 'fallback_combined'
                result['fallback_used']= True
                result['fallback_note']= (
                    f'Low confidence ({result["confidence"]:.0%}) on ensemble. '
                    f'Re-validated with {method_tag} and blended (60/40 weighted).'
                )
                result['models_used'] = result.get('models_used', []) + fb_result.get('models_used', [])
        else:
            if low_confidence:
                result['fallback_note'] = (
                    'Low confidence detected. Backup APIs unreachable or unconfigured.'
                )

    # Ensure required keys exist
    result.setdefault('method',       'ensemble')
    result.setdefault('fallback_used', False)
    result.setdefault('fallback_note', '')
    return result


# ── Background retraining ──────────────────────────────────────────────────────

retrain_lock = threading.Lock()

def run_retrain(job_id: int, cfg: dict, dataset_path: str,
                output_path: str, db_update_fn):
    """
    Execute in a daemon thread.  Trains a new model with the given config
    and saves it to output_path.

    db_update_fn(job_id, status, log_line, finished) is called for progress.
    """
    def log_line(text: str, status: str = None):
        msg = f"[{time.strftime('%H:%M:%S')}] {text}\n"
        db_update_fn(job_id,
                     status   = status,
                     log_append = msg,
                     finished = (status in ('done', 'failed')))
        log.info("[retrain job %d] %s", job_id, text)

    with retrain_lock:
        try:
            log_line("Starting training job …", 'running')
            tf = _get_tf()
            from tensorflow.keras import layers, Model
            from tensorflow.keras.callbacks import (EarlyStopping,
                                                     ModelCheckpoint,
                                                     ReduceLROnPlateau)
            from tensorflow.keras.preprocessing.image import ImageDataGenerator

            img_size  = int(cfg.get('image_size', 224))
            epochs    = int(cfg.get('epochs', 10))
            lr        = float(cfg.get('lr', 1e-4))
            batch     = int(cfg.get('batch_size', 32))
            dropout   = float(cfg.get('dropout', 0.3))
            dense_u   = int(cfg.get('dense_units', 256))
            base_name = cfg.get('base_model', 'MobileNetV2')

            if not os.path.isdir(dataset_path):
                log_line(f"Dataset path not found: {dataset_path}", 'failed')
                return

            # Data generators
            log_line("Building data generators …")
            train_gen = ImageDataGenerator(
                rescale=1./255, rotation_range=20,
                width_shift_range=0.1, height_shift_range=0.1,
                horizontal_flip=True, validation_split=0.2
            )
            train_data = train_gen.flow_from_directory(
                dataset_path, target_size=(img_size, img_size),
                batch_size=batch, class_mode='categorical', subset='training'
            )
            val_data = train_gen.flow_from_directory(
                dataset_path, target_size=(img_size, img_size),
                batch_size=batch, class_mode='categorical', subset='validation'
            )
            n_classes = train_data.num_classes
            log_line(f"Classes: {n_classes}, train={train_data.samples}, val={val_data.samples}")

            # Base model
            base_map = {
                'MobileNetV2':  tf.keras.applications.MobileNetV2,
                'DenseNet121':  tf.keras.applications.DenseNet121,
                'EfficientNetB3': tf.keras.applications.EfficientNetB3,
                'ResNet50V2':   tf.keras.applications.ResNet50V2,
                'VGG16':        tf.keras.applications.VGG16,
            }
            BaseModel = base_map.get(base_name, tf.keras.applications.MobileNetV2)
            log_line(f"Building model: {base_name} …")
            base = BaseModel(include_top=False,
                             weights='imagenet',
                             input_shape=(img_size, img_size, 3))
            base.trainable = False

            x = base.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(dense_u, activation='relu')(x)
            x = layers.Dropout(dropout)(x)
            out = layers.Dense(n_classes, activation='softmax')(x)
            model = Model(inputs=base.input, outputs=out)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0),
            ]

            log_line(f"Training for up to {epochs} epochs …")
            for ep in range(epochs):
                hist = model.fit(
                    train_data, validation_data=val_data,
                    epochs=1, callbacks=[], verbose=0
                )
                acc    = hist.history['accuracy'][0]
                v_acc  = hist.history['val_accuracy'][0]
                log_line(f"  Epoch {ep+1}/{epochs} — acc={acc:.4f}  val_acc={v_acc:.4f}")

            model.save(output_path)
            log_line(f"Model saved to: {output_path}", 'done')

        except Exception:
            log_line("Training failed:\n" + traceback.format_exc(), 'failed')
