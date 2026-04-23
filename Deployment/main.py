import numpy as np
from scipy import signal as signal_sci
from scipy.stats import skew, kurtosis
import matplotlib.mlab as mlab
import math
import json
import os
import time
import onnxruntime as ort
from sdr_stream import SdrStream

def convert_float_to_int8(signal_float):
    """Convert float signal to int8 using the same normalization as dataset generation"""
    signal_min = signal_float.min()
    signal_max = signal_float.max()
    if signal_max == signal_min:
        return np.zeros_like(signal_float, dtype=np.int8)
    signal_int8 = ((signal_float - signal_min) / (signal_max - signal_min) * 255 - 128).astype(np.int8)
    return signal_int8

def extract_features_direct(raw_signal, fs=10e6):
    """
    Extract 16 features directly from in-memory signal.
    Uses same logic as signal_analysis.extract_features() but works on raw signal.
    """
    real_signal = raw_signal.real if np.iscomplexobj(raw_signal) else raw_signal
    
    mean_ = np.mean(real_signal)
    median_ = np.median(real_signal)
    std_ = np.std(real_signal)
    mad_ = np.mean(np.absolute(real_signal - np.mean(real_signal)))
    rms_ = np.sqrt(np.mean(real_signal ** 2))
    percentile_25th_ = np.quantile(real_signal, 0.25)
    percentile_75th_ = np.quantile(real_signal, 0.75)
    iqr_ = np.subtract(*np.percentile(real_signal, [75, 25]))
    skewness_ = skew(real_signal)
    kurtosis_ = kurtosis(real_signal, fisher=False, bias=True)
    
    # Frequency domain features
    nfft = 512
    win = signal_sci.get_window('boxcar', nfft)
    signal_centered = real_signal - np.mean(real_signal)
    freq, psd = signal_sci.welch(signal_centered, fs=fs, nfft=nfft, window=win, 
                                 noverlap=0, return_onesided=False)
    max_power_idx = np.argmax(psd)
    max_power_win_ = psd[max_power_idx]
    freq_max_power_ = freq[max_power_idx]
    mean_power_win_ = np.mean(psd)
    
    # Simplified entropy and pentropy (can be enhanced with full implementations)
    entropy_ = 0
    pentropy_mean_ = 0
    pentropy_std_ = 0
    
    features_list = [mean_, median_, std_, mad_, rms_, percentile_25th_, percentile_75th_,
                     iqr_, skewness_, kurtosis_, entropy_, max_power_win_, freq_max_power_,
                     mean_power_win_, pentropy_mean_, pentropy_std_]
    return features_list

def compute_spectrogram(signal_int8, fs=10e6, nfft=128, overlap_percentage=0.999):
    """
    Compute spectrogram using same parameters as signal_analysis.spectrogram_image()
    Returns spectrogram array (nfft, time_steps)
    """
    window = ('kaiser', 5.0)
    win = signal_sci.get_window(window, nfft)
    number_overlap = math.floor(nfft * overlap_percentage)
    
    spec, freq, t = mlab.specgram(x=signal_int8,
                                  Fs=fs,
                                  NFFT=nfft,
                                  window=win,
                                  noverlap=number_overlap)
    return spec, freq, t


def _resize_2d_nearest(array_2d, target_h, target_w):
    """Simple nearest-neighbor resize for 2D arrays using NumPy indexing."""
    src_h, src_w = array_2d.shape
    if src_h == target_h and src_w == target_w:
        return array_2d

    row_idx = np.clip(np.round(np.linspace(0, src_h - 1, target_h)).astype(np.int64), 0, src_h - 1)
    col_idx = np.clip(np.round(np.linspace(0, src_w - 1, target_w)).astype(np.int64), 0, src_w - 1)
    return array_2d[row_idx][:, col_idx]


class OnnxLiveClassifier:
    def __init__(self, onnx_path, class_names_path=None):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()

        if len(self.inputs) < 2:
            raise RuntimeError("Expected at least 2 model inputs (spectrogram and features)")

        # Detect input roles from rank: spectrogram is 4D, features are 2D.
        signal_input = None
        feature_input = None
        for inp in self.inputs:
            shape = inp.shape
            rank = len(shape) if shape is not None else 0
            if rank == 4 and signal_input is None:
                signal_input = inp
            elif rank == 2 and feature_input is None:
                feature_input = inp

        if signal_input is None or feature_input is None:
            # Fallback to input order if rank-based detection is not possible.
            signal_input = self.inputs[0]
            feature_input = self.inputs[1]

        self.signal_input_name = signal_input.name
        self.feature_input_name = feature_input.name

        # Parse expected spatial and feature dimensions when statically available.
        self.expected_h = None
        self.expected_w = None
        self.expected_feature_dim = None

        signal_shape = signal_input.shape
        feature_shape = feature_input.shape

        if len(signal_shape) == 4:
            if isinstance(signal_shape[2], int):
                self.expected_h = signal_shape[2]
            if isinstance(signal_shape[3], int):
                self.expected_w = signal_shape[3]

        if len(feature_shape) == 2 and isinstance(feature_shape[1], int):
            self.expected_feature_dim = feature_shape[1]

        self.class_names = None
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)

        print("--- ONNX session initialized")
        print(f"--- Input names: signal='{self.signal_input_name}', features='{self.feature_input_name}'")
        print(f"--- Output names: {[o.name for o in self.outputs]}")

    def _prepare_spectrogram(self, spec):
        spec = np.asarray(spec, dtype=np.float32)

        # Power spectrogram can contain very small values; log compression improves numeric stability.
        spec = np.log1p(np.maximum(spec, 0.0))

        if self.expected_h is not None and self.expected_w is not None:
            spec = _resize_2d_nearest(spec, self.expected_h, self.expected_w)

        return spec[np.newaxis, np.newaxis, :, :].astype(np.float32)

    def _prepare_features(self, features):
        feat = np.asarray(features, dtype=np.float32)
        feat = feat.reshape(-1)

        if self.expected_feature_dim is not None:
            if feat.shape[0] < self.expected_feature_dim:
                pad = self.expected_feature_dim - feat.shape[0]
                feat = np.pad(feat, (0, pad), mode="constant")
            elif feat.shape[0] > self.expected_feature_dim:
                feat = feat[:self.expected_feature_dim]

        return feat[np.newaxis, :].astype(np.float32)

    def predict(self, spec, features):
        signal_input = self._prepare_spectrogram(spec)
        feature_input = self._prepare_features(features)

        feed = {
            self.signal_input_name: signal_input,
            self.feature_input_name: feature_input,
        }

        out_names = [o.name for o in self.outputs]
        out_vals = self.session.run(out_names, feed)

        # By export convention, first output is logits.
        logits = np.asarray(out_vals[0], dtype=np.float32)
        if logits.ndim != 2:
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

        logits_row = logits[0]
        pred_idx = int(np.argmax(logits_row))

        # Stable softmax for confidence.
        shifted = logits_row - np.max(logits_row)
        probs = np.exp(shifted)
        probs = probs / np.sum(probs)
        confidence = float(probs[pred_idx])

        pred_label = str(pred_idx)
        if self.class_names is not None and pred_idx < len(self.class_names):
            pred_label = self.class_names[pred_idx]

        energy = None
        for out in out_vals[1:]:
            arr = np.asarray(out)
            if arr.ndim == 1 and arr.shape[0] == 1:
                energy = float(arr[0])
                break

        return {
            "pred_idx": pred_idx,
            "pred_label": pred_label,
            "confidence": confidence,
            "energy": energy,
        }

def main():
    onnx_path = "./Deployment/export/jamming_model.onnx"
    class_names_path = "./Deployment/export/class_names.json"

    classifier = OnnxLiveClassifier(onnx_path=onnx_path, class_names_path=class_names_path)

    sdr = SdrStream()
    fs = 10e6  # Sampling rate (HackRF One)
    
    # Start streaming from HackRF One
    sdr.start_stream()
    
    # Give the streaming thread time to start and begin collecting data
    import time as time_module
    print("--- Waiting for SDR streaming thread to initialize...")
    time_module.sleep(2)
    
    if not sdr.flowgraph_started.is_set():
        print("ERROR: SDR streaming thread did not start!")
    else:
        print("--- SDR streaming thread confirmed started")

    try:
        iteration = 0
        while True:
            iteration += 1
            # Get complex64 IQ samples from buffer
            iq_samples = sdr.get_samples(window_chunk=200e-6)
            
            if iteration % 20 == 0:
                print(f"[Iteration {iteration}] Buffer size: {len(sdr.iq_buffer)}, Last IQ shape: {iq_samples.shape if iq_samples is not None else 'None'}")
            
            if iq_samples is None:
                continue
            
            # Convert complex64 IQ to magnitude (float32) - substitute the gnu radio block
            signal_float = np.abs(iq_samples).astype(np.float32)
            
            # Convert to int8 using same method as offline dataset generation
            signal_int8 = convert_float_to_int8(signal_float)
            
            # Compute spectrogram
            spec, freq, t = compute_spectrogram(signal_int8, fs=fs)
            
            # Extract features directly from signal
            features = extract_features_direct(signal_int8, fs)

            # normalize features using same method as offline dataset generation by loading pkl scaler and applying transform
            scaler_path = "./Deployment/export/scaler_model.pkl"
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Feature scaler not found: {scaler_path}")
            
            # Try joblib first (more robust), fall back to pickle
            try:
                import joblib
                scaler = joblib.load(scaler_path)
            except:
                import pickle
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            
            features_scaled = scaler.transform([features])[0]

            prediction = classifier.predict(spec=spec, features=features_scaled)

            energy_msg = "n/a"
            if prediction["energy"] is not None:
                energy_msg = f"{prediction['energy']:.4f}"

            print(
                "Pred: {label} (idx={idx}) | conf={conf:.3f} | energy={energy} | spec={shape}".format(
                    label=prediction["pred_label"],
                    idx=prediction["pred_idx"],
                    conf=prediction["confidence"],
                    energy=energy_msg,
                    shape=spec.shape,
                )
            )

            # Optional tiny sleep to keep console readable.
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("Stopping signal acquisition...")
    finally:
        sdr.stop_stream()


if __name__ == "__main__":
    main()