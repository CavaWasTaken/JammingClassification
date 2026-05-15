import pandas as pd
import numpy as np
import json
import os
import time
import onnxruntime as ort
from sdr_stream import SdrStream
from signal_analysis_live import SignalAnalysisLive
from osr_stats import load_osr_stats, evaluate_osr
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

def convert_float_to_int8(signal_float):
    """Convert float signal to int8 using the same normalization as dataset generation"""
    signal_min = signal_float.min()
    signal_max = signal_float.max()
    if signal_max == signal_min:
        return np.zeros_like(signal_float, dtype=np.int8)
    signal_int8 = ((signal_float - signal_min) / (signal_max - signal_min) * 255 - 128).astype(np.int8)
    return signal_int8

def main():
    onnx_file_path = "./Deployment/export/jamming_model.onnx"
    class_names_path = "./Deployment/export/class_names.json"

    sess = ort.InferenceSession(onnx_file_path)

    with open(class_names_path, "r") as f:
        class_names = json.load(f)

    osr_stats_path = "./Deployment/export/osr_stats.npz"
    osr_stats = None
    if os.path.exists(osr_stats_path):
        osr_stats = load_osr_stats(osr_stats_path)
        print(f"--- Loaded OSR stats from: {osr_stats_path}")
    else:
        print(f"WARNING: OSR stats not found at {osr_stats_path}. Live realism check disabled.")

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

    # create a csv file where to write predictions with columns: timestamp, pred_label, confidence, energy
    csv_path = "./Deployment/predictions.csv"
    with open(csv_path, "w") as f:
        f.write(
            "timestamp,pred_label,osr_label,osr_realistic,osr_distance,osr_threshold,"
            "confidence,probability,margin,energy,spectrogram_time_ms,int8_conversion_time_ms,"
            "feature_extraction_time_ms,feature_scaling_time_ms,processing_time,inference_time_ms\n"
        )

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

    signal_analysis_live_object = SignalAnalysisLive(fs=fs)    # create the SignalAnalysis object

    try:
        iteration = 0
        while True:
            iteration += 1
            # Get complex64 IQ samples from buffer
            iq_samples = sdr.get_samples(window_chunk=200e-6)

            start_time_processing = time.perf_counter()
            
            if iteration % 20 == 0:
                print(f"[Iteration {iteration}] Buffer size: {len(sdr.iq_buffer)}, Last IQ shape: {iq_samples.shape if iq_samples is not None else 'None'}")
            
            if iq_samples is None:
                continue

            # iq_samples is complex64 array of shape (num_samples,), where num_samples = window_chunk * fs = 200e-6 * 10e6 = 2000 samples per chunk

            # for computing the spectogrm use iq_samples with 1000 samples to match the dataset parameters
            iq_samples_ = iq_samples[:1000]

            start_time_spectrogram = time.perf_counter()

            # Compute spectrogram
            spec, freq, t = signal_analysis_live_object.compute_spectrogram(iq_samples_)

            end_time_spectrogram = time.perf_counter()
            spectrogram_time_ms = (end_time_spectrogram - start_time_spectrogram) * 1000

            # show the spectrogram shape and value range for debugging
            # print(f"\n--- Spectrogram shape: {spec.shape}, value range: [{spec.min()}, {spec.max()}]")
            # print(f"\n--- Frequency bins: {len(freq)}, Time bins: {len(t)}")
            # print(f"\n--- Frequency range: [{freq.min()}, {freq.max()}], Time range: [{t.min()}, {t.max()}]")

            start_time_int8_conversion = time.perf_counter()

            # convert the spectogram to int8
            spec_int8 = convert_float_to_int8(spec)

            end_time_int8_conversion = time.perf_counter()
            int8_conversion_time_ms = (end_time_int8_conversion - start_time_int8_conversion) * 1000

            # print(f"\n--- Spectrogram int8 shape: {spec_int8.shape}, value range: [{spec_int8.min()}, {spec_int8.max()}]")

            start_time_feature_extraction = time.perf_counter()

            # Extract features directly from signal
            features = signal_analysis_live_object.extract_features_direct(iq_samples)

            end_time_feature_extraction = time.perf_counter()
            feature_extraction_time_ms = (end_time_feature_extraction - start_time_feature_extraction) * 1000

            # print(f'\n--- Number of features extracted: {len(features)}')
            # print(f"\n--- Extracted features:\n{features}")

            # feature_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16']

            # features_df = pd.DataFrame([features], columns=feature_names)

            # normalize features using same method as offline dataset generation by loading pkl scaler and applying transform

            start_time_feature_scaling = time.perf_counter()

            features_scaled = scaler.transform([features])[0]

            end_time_feature_scaling = time.perf_counter()
            feature_scaling_time_ms = (end_time_feature_scaling - start_time_feature_scaling) * 1000

            # print(f'\n--- Number of features after scaling: {len(features_scaled)}')
            # print(f"\n--- Scaled features:\n{features_scaled}")

            # write the scaled features into a csv file for debugging
            # features_csv_path = "./Deployment/extracted_features.csv"
            # with open(features_csv_path, "a") as f:               
            #     f.write(",".join(map(str, features_scaled)) + "\n")

            end_time_processing = time.perf_counter()

            processing_time_ms = (end_time_processing - start_time_processing) * 1000

            inputs = {
                'spectrogram': spec_int8[np.newaxis, np.newaxis, :, :].astype(np.float32),
                'features': features_scaled[np.newaxis, :].astype(np.float32),
            }

            start_time = time.perf_counter()

            outputs = sess.run(None, inputs)

            logits = outputs[0]
            penultimate = outputs[1]
            energy = outputs[2]

            print('\n--- Model inference completed ---')

            end_time = time.perf_counter()

            inference_time_ms = (end_time - start_time) * 1000
            fps = 1.0 / (end_time - start_time)

            predicted_class = np.argmax(logits[0])
            class_predicted_name = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"

            if osr_stats is not None:
                osr_eval = evaluate_osr(penultimate[0], class_predicted_name, osr_stats)
                osr_label = osr_eval['osr_label']
                osr_distance = osr_eval['distance']
                osr_threshold = osr_eval['threshold']
                osr_realistic = int(osr_eval['is_realistic'])
            else:
                osr_label = "UNKNOWN"
                osr_distance = float('inf')
                osr_threshold = float('inf')
                osr_realistic = 0
            # print(f"\n--- Predicted class: {class_predicted_name} | Inference time: {inference_time_ms:.2f} ms | FPS: {fps:.1f}")
            # print(f"\n--- Logits: {logits}")
            # print(f"\n--- Energy: {energy}")
            # print(f"\n--- Penultimate shape: {penultimate.shape}")

            # apply softmax to logits to get probabilities
            exp_logits = np.exp(logits[0] - np.max(logits[0]))
            probabilities = exp_logits / np.sum(exp_logits)

            # compute the margin between the top 2 predicted classes as a confidence measure
            sorted_logits = np.sort(logits[0])
            margin = sorted_logits[-1] - sorted_logits[-2]

            csv_path = "./Deployment/predictions.csv"
            timestamp = time.time()
            with open(csv_path, "a") as f:
                f.write(
                    f"{timestamp},{class_predicted_name},{osr_label},{osr_realistic},{osr_distance:.4f},{osr_threshold:.4f},"
                    f"{np.max(logits[0]):.4f},{np.max(probabilities):.4f},{energy[0]:.4f},{margin:.4f},"
                    f"{spectrogram_time_ms:.2f},{int8_conversion_time_ms:.2f},{feature_extraction_time_ms:.2f},"
                    f"{feature_scaling_time_ms:.2f},{processing_time_ms:.2f},{inference_time_ms:.2f}\n"
                )

            print(
                f"Pred: {class_predicted_name} | OSR: {osr_label} | realistic={osr_realistic} "
                f"(dist={osr_distance:.3f}, thr={osr_threshold:.3f})"
            )

            # Optional tiny sleep to keep console readable.
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("Stopping signal acquisition...")
    finally:
        sdr.stop_stream()


if __name__ == "__main__":
    main()