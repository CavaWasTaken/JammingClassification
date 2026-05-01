import pandas as pd
import numpy as np
import json
import os
import time
import onnxruntime as ort
from sdr_stream import SdrStream
from signal_analysis_live import SignalAnalysisLive
import warnings
import paho.mqtt.client as mqtt
import ssl

warnings.filterwarnings('ignore', message='X does not have valid feature names')

def convert_float_to_int8(signal_float):
    """Convert float signal to int8 using the same normalization as dataset generation"""
    signal_min = signal_float.min()
    signal_max = signal_float.max()
    if signal_max == signal_min:
        return np.zeros_like(signal_float, dtype=np.int8)
    signal_int8 = ((signal_float - signal_min) / (signal_max - signal_min) * 255 - 128).astype(np.int8)
    return signal_int8

def check_file_exists(filepath, description):
    """To verify that the required file exists before proceeding"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: {description} not found at path: {filepath}")

def main():
    # --- PRE-FLIGHT CHECKS ---
    config_path = 'config.json'
    onnx_file_path = "./Deployment/export/jamming_model.onnx"
    class_names_path = "./Deployment/export/class_names.json"
    scaler_path = "./Deployment/export/scaler_model.pkl"

    check_file_exists(config_path, "Configuration file")
    check_file_exists(onnx_file_path, "ONNX model file")
    check_file_exists(class_names_path, "Class names JSON file")
    check_file_exists(scaler_path, "Scaler model file")

    LATITUDE = 45.0621
    LONGITUDE = 7.6622

    # --- LOAD CONFIG & STATIC DATA  ---
    with open(config_path, 'r') as f:
        config = json.load(f)

    with open(class_names_path, "r") as f:
        class_names = json.load(f)

    hive_credentials = config["hivemq"]
    topics = config["topics"]

    broker = hive_credentials["host"]
    port = hive_credentials["port_tcp"]
    user = hive_credentials["username"]
    password = hive_credentials["password"]

    TRUCK_ID = config["plate"]
    
    # --- SETUP MQTT ---
    client = mqtt.Client(f"Truck-{TRUCK_ID}")
    client.tls_set(cert_reqs=ssl.CERT_NONE) 
    client.username_pw_set(user, password)

    topic_alert = topics["alerts"] + f"/{TRUCK_ID}"
    topic_status = topics["status"] + f"/{TRUCK_ID}"

    client.will_set(
        topic=topic_status,
        payload=json.dumps({
                "truckId": TRUCK_ID,
                "status": "OFFLINE",
                "timestamp": time.time(),
                "location": {"latitude": LATITUDE, "longitude": LONGITUDE} 
            }),
        qos=1,
        retain=True
    )

    def on_connect(client, userdata, flags, rc):
        print(f"- Truck Connected to Cloud (Code: {rc})")

    client.on_connect = on_connect

    try:
        client.connect(broker, port)
        client.loop_start() # Run network loop in background
    except Exception as e:
        print(f"ERROR: Cannot connect to MQTT broker ({e}).")

    # --- LOAD MODELS ---
    print("--- Loading Models...")
    sess = ort.InferenceSession(onnx_file_path)

    try:
        import joblib
        scaler = joblib.load(scaler_path)
    except ImportError:
        import pickle
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # --- SETUP SDR ---
    sdr = SdrStream()
    fs = 10e6  # Sampling rate (HackRF One)
    signal_analysis_live_object = SignalAnalysisLive(fs=fs)
    
    sdr.start_stream()
    
    print("--- Waiting for SDR streaming thread to initialize...")
    time.sleep(2)
    
    if not sdr.flowgraph_started.is_set():
        print("ERROR: SDR streaming thread did not start!")
        return # Esce se l'SDR non parte
    else:
        print("--- SDR streaming thread confirmed started")

    # --- SETUP CSV ---
    csv_path = "./Deployment/predictions.csv"
    
    f_csv = open(csv_path, "w")
    f_csv.write("timestamp,pred_label,confidence,probability,energy,spectrogram_time_ms,int8_conversion_time_ms,feature_extraction_time_ms,feature_scaling_time_ms,processing_time,inference_time_ms\n")

    try:
        iteration = 0
        while True:
            iteration += 1
            
            try:
                # Get complex64 IQ samples from buffer
                iq_samples = sdr.get_samples(window_chunk=200e-6)
                start_time_processing = time.perf_counter()
                
                if iteration % 20 == 0:
                    print(f"[Iteration {iteration}] Buffer size: {len(sdr.iq_buffer)}, Last IQ shape: {iq_samples.shape if iq_samples is not None else 'None'}")
                
                if iq_samples is None or len(iq_samples) < 1000:
                    continue

                iq_samples_ = iq_samples[:1000]


                # --- 1. SPECTROGRAM ---
                start_time_spectrogram = time.perf_counter()
                spec, freq, t = signal_analysis_live_object.compute_spectrogram(iq_samples_)
                spectrogram_time_ms = (time.perf_counter() - start_time_spectrogram) * 1000

                # --- 2. INT8 CONVERSION ---
                start_time_int8_conversion = time.perf_counter()
                spec_int8 = convert_float_to_int8(spec)
                int8_conversion_time_ms = (time.perf_counter() - start_time_int8_conversion) * 1000

                # --- 3. FEATURE EXTRACTION ---
                start_time_feature_extraction = time.perf_counter()
                features = signal_analysis_live_object.extract_features_direct(iq_samples)
                feature_extraction_time_ms = (time.perf_counter() - start_time_feature_extraction) * 1000

                # --- 4. FEATURE SCALING ---
                start_time_feature_scaling = time.perf_counter()
                features_scaled = scaler.transform([features])[0]
                feature_scaling_time_ms = (time.perf_counter() - start_time_feature_scaling) * 1000

                processing_time_ms = (time.perf_counter() - start_time_processing) * 1000

                print("--- Pre-Processing completed ---")

                # --- 5. INFERENCE ---
                inputs = {
                    'spectrogram': spec_int8[np.newaxis, np.newaxis, :, :].astype(np.float32),
                    'features': features_scaled[np.newaxis, :].astype(np.float32),
                }

                start_time = time.perf_counter()
                outputs = sess.run(None, inputs)
                inference_time_ms = (time.perf_counter() - start_time) * 1000

                logits = outputs[0]
                penultimate = outputs[1]
                energy = outputs[2]

                # --- 6. POST-PROCESSING & ALERTING ---
                predicted_class = np.argmax(logits[0])
                class_predicted_name = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"
                
                if class_predicted_name not in ["CLEAN", "Unknown"]:
                    parts = class_predicted_name.split("_")
                    jamming_type = parts[0] if len(parts) > 0 else "Unknown"
                    jamming_level = parts[1] if len(parts) > 1 else "Unknown"
                    
                    payload = {
                        "truckId": TRUCK_ID,
                        "type": jamming_type,
                        "level": jamming_level,
                        "timestamp": time.time(),
                        "location": {"latitude": LATITUDE, "longitude": LONGITUDE} 
                    }
                
                    client.publish(topic_alert, json.dumps(payload))
                    print(f"- JAMMING DETECTED {class_predicted_name} - Alert sent to Cloud")

                # Added 1e-9 to avoid division by zero in case of very low confidence
                probability = np.max(logits[0]) / (np.sum(logits[0]) + 1e-9)

                # --- 7. CSV WRITING ---
                f_csv.write(f"{time.time()},{class_predicted_name},{np.max(logits[0]):.4f},{probability:.4f},{energy[0]:.4f},{spectrogram_time_ms:.2f},{int8_conversion_time_ms:.2f},{feature_extraction_time_ms:.2f},{feature_scaling_time_ms:.2f},{processing_time_ms:.2f},{inference_time_ms:.2f}\n")
                
                # Facciamo il flush ogni tanto per assicurarci che i dati vengano scritti su disco, 
                # pur mantenendo le performance (es. ogni 20 iterazioni)
                if iteration % 20 == 0:
                    f_csv.flush()

                time.sleep(0.05)
            
            except Exception as e:
                print(f"Error during iteration {iteration}: {e}")

    except KeyboardInterrupt:
        print("Stopping signal acquisition...")
    finally:
        sdr.stop_stream()
        f_csv.close() 
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()