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
import threading
import serial
import serial.tools.list_ports
from pyubx2 import UBXReader, UBX_PROTOCOL

warnings.filterwarnings('ignore', message='X does not have valid feature names')



class UbloxGpsThread(threading.Thread): 
    # con questa classe CHAT mi dice che il file ubx_position_parser.py è troppo pesante 
    # per essere eseguito in tempo reale, quindi ho integrato direttamente la lettura del
    #  GPS in questo thread ottimizzato, che si occupa solo di leggere i dati GPS e
    #  aggiornare le coordinate in background. 
    # In questo modo evitiamo il sovraccarico di importare un modulo esterno e possiamo 
    # gestire tutto internamente in modo più efficiente.
    def __init__(self):
        super().__init__()
        self.latitude = 41.9
        self.longitude = 12.5
        self.fix_type = 0
        self.running = True
        self.port = self.find_ublox_port()

    def find_ublox_port(self):
        """Automatically searches for the port of the u-blox module"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            desc = port.description or ""
            manuf = port.manufacturer or ""
            if "u-blox" in desc or "u-blox" in manuf:
                return port.device
            if (port.vid, port.pid) in [(0x1546, 0x01A7)]:
                return port.device
        return None

    def run(self):
        """This is the loop that runs in the background"""
        if not self.port:
            print("GPS: u-blox module not found. Fixed coordinates will be used.")
            return

        print(f"GPS: Connected to the {self.port} port. Background reading started.")
        try:
            with serial.Serial(self.port, baudrate=230400, timeout=1) as ser:
                ubr = UBXReader(ser, protfilter=UBX_PROTOCOL)
                while self.running:
                    raw, parsed = ubr.read()
                    if parsed is None:
                        continue
                    
                    # If the message is NAV-PVT, we update the position variables
                    if parsed.identity == "NAV-PVT":
                        self.fix_type = getattr(parsed, "fixType", 0)
                        
                        # We update the coordinates only if we have a valid fix (2D, 3D or GNSS+DR)
                        if self.fix_type >= 2:
                            self.latitude = getattr(parsed, "lat", self.latitude)
                            self.longitude = getattr(parsed, "lon", self.longitude)
                            
        except Exception as e:
            print(f"Error in GPS thread: {e}")

    def stop(self):
        """Stop the thread cleanly"""
        self.running = False

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

STATE_FILE = "./Deployment/export/last_known_location.json"

def load_last_location(default_lat=41.9, default_lon=12.5):
    """Load the last known location from the local file."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                return data.get("latitude", default_lat), data.get("longitude", default_lon)
    except Exception as e:
        print(f"Failed to load last location: {e}")
    return default_lat, default_lon

def save_last_location(lat, lon):
    """Save the last known location to the local file."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"latitude": lat, "longitude": lon}, f)
    except Exception as e:
        print(f"Failed to save last location: {e}")

def main():
    # --- PRE-FLIGHT CHECKS ---
    config_path = './Deployment/export/config.json'
    onnx_file_path = "./Deployment/export/jamming_model.onnx"
    class_names_path = "./Deployment/export/class_names.json"
    scaler_path = "./Deployment/export/scaler_model.pkl"

    check_file_exists(config_path, "Configuration file")
    check_file_exists(onnx_file_path, "ONNX model file")
    check_file_exists(class_names_path, "Class names JSON file")
    check_file_exists(scaler_path, "Scaler model file")

    # --- START THE GPS THREAD ---
    gps_tracker = UbloxGpsThread()
    gps_tracker.daemon = True # So the thread closes itself when you close the programa
    gps_tracker.start()

    last_clean_latitude, last_clean_longitude = load_last_location()
    print(f"--- Initial position loaded: Lat {last_clean_latitude}, Lon {last_clean_longitude}")

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
                "location": {"latitude": last_clean_latitude, "longitude": last_clean_longitude} 
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
    
    # Let's check what is actually installed/available on the system
    available_providers = ort.get_available_providers()
    
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_max_workspace_size': 2147483648,
            'trt_fp16_enable': True,
        }),
        ('CUDAExecutionProvider', {
            'device_id': 0,
        })
    ]
    
    
    if 'TensorrtExecutionProvider' in available_providers:
        sess = ort.InferenceSession(onnx_file_path, providers=providers)
        print("ONNX successfully loaded with GPU acceleration.")
    else:
        print("GPU providers not found, falling back to CPU.")
        sess = ort.InferenceSession(onnx_file_path, providers=['CPUExecutionProvider'])

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
        return # Exits if the SDR doesn't start
    else:
        print("--- SDR streaming thread confirmed started")

    # --- SETUP CSV ---
    csv_path = "./Deployment/predictions.csv"
    
    f_csv = open(csv_path, "w")
    f_csv.write("index,timestamp,pred_label,confidence,probability,energy,spectrogram_time_ms,int8_conversion_time_ms,feature_extraction_time_ms,feature_scaling_time_ms,processing_time,inference_time_ms\n")

    # STATUS_INTERVAL = 10 # Keep-alive interval for status updates (in seconds)
    STATUS_INTERVAL = config["status_update_interval_seconds"]
    last_status_time = time.time()
    ALERT_THRESHOLD = config["number_of_consecutive_alerts"]

    try:
        iteration = 0
        # --- DEBOUNCE LOGIC FOR ALERT ---
        consecutive_jamming_count = 0
        last_jamming_type = None
        # ALERT_THRESHOLD = 3  # Number of consecutive measurements required
        
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
                #spec, _, _ = signal_analysis_live_object.compute_spectrogram_general(iq_samples_) # Compute spectrogram using GPU if available, CPU otherwise
                spectrogram_time_ms = (time.perf_counter() - start_time_spectrogram) * 1000

                # --- 2. INT8 CONVERSION ---
                start_time_int8_conversion = time.perf_counter()
                spec_int8 = convert_float_to_int8(spec)
                int8_conversion_time_ms = (time.perf_counter() - start_time_int8_conversion) * 1000

                # --- 3. FEATURE EXTRACTION ---
                start_time_feature_extraction = time.perf_counter()
                features = signal_analysis_live_object.extract_features_direct(iq_samples)
                #features = signal_analysis_live_object.extract_features_direct_optimized(iq_samples) #faster
                feature_extraction_time_ms = (time.perf_counter() - start_time_feature_extraction) * 1000

                # --- 4. FEATURE SCALING ---
                start_time_feature_scaling = time.perf_counter()
                features_scaled = scaler.transform([features])[0]
                feature_scaling_time_ms = (time.perf_counter() - start_time_feature_scaling) * 1000

                processing_time_ms = (time.perf_counter() - start_time_processing) * 1000

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
                
                if class_predicted_name == "CLEAN":
                    last_clean_latitude = gps_tracker.latitude
                    last_clean_longitude = gps_tracker.longitude
                    # Reset the counter if the signal becomes clear again
                    consecutive_jamming_count = 0
                    last_jamming_type = None
                                
                elif class_predicted_name != "Unknown":
                    # Confirmation logic: must be the SAME type consecutively
                    if class_predicted_name == last_jamming_type:
                        consecutive_jamming_count += 1
                    else:
                        last_jamming_type = class_predicted_name
                        consecutive_jamming_count = 1
                    
                    # Sending alert only when the threshold is reached
                    if consecutive_jamming_count == ALERT_THRESHOLD:
                        # JAMMING DETECTED
                        parts = class_predicted_name.split("_")
                        jamming_type = parts[0] if len(parts) > 0 else "Unknown"
                        jamming_level = parts[1] if len(parts) > 1 else "Unknown"

                        payload = {
                            "truckId": TRUCK_ID,
                            "type": jamming_type,
                            "level": jamming_level,
                            "timestamp": time.time(),
                            "location": {
                                "latitude": last_clean_latitude,
                                "longitude": last_clean_longitude
                            }
                        }
                    
                        client.publish(topic_alert, json.dumps(payload))
                        print(f"- JAMMING DETECTED ({jamming_type}) - Alert sent to Cloud at Last Known Good Location (Lat: {last_clean_latitude}, Lon: {last_clean_longitude})")

                # Added 1e-9 to avoid division by zero in case of very low confidence
                probability = np.max(logits[0]) / (np.sum(logits[0]) + 1e-9)

                # --- 7. CSV WRITING ---
                f_csv.write(f"{iteration},{time.time()},{class_predicted_name},{np.max(logits[0]):.4f},{probability:.4f},{energy[0]:.4f},{spectrogram_time_ms:.2f},{int8_conversion_time_ms:.2f},{feature_extraction_time_ms:.2f},{feature_scaling_time_ms:.2f},{processing_time_ms:.2f},{inference_time_ms:.2f}\n")
                
                # Facciamo il flush ogni tanto per assicurarci che i dati vengano scritti su disco, 
                # pur mantenendo le performance (es. ogni 20 iterazioni)
                if iteration % 5 == 0:
                    f_csv.flush()

                # --- 8. STATUS UPDATE ---
                current_time = time.time()
                if current_time - last_status_time >= STATUS_INTERVAL:
                    payload_status = {
                        "truckId": TRUCK_ID,
                        "status": "ACTIVE",
                        "timestamp": current_time,
                        "location": {
                            "latitude": last_clean_latitude, 
                            "longitude": last_clean_longitude
                        }
                    }
                    client.publish(topic_status, json.dumps(payload_status))
                    print(f"- STATUS sent (Lat: {last_clean_latitude}, Lon: {last_clean_longitude})")

                    save_last_location(last_clean_latitude, last_clean_longitude)
                    last_status_time = current_time

                time.sleep(0.05)
            
            except Exception as e:
                print(f"Error during iteration {iteration}: {e}")

    except KeyboardInterrupt:
        print("Stopping signal acquisition...")
    finally:
        #     # --- SEND OFFLINE STATUS MANUALLY ---
        # print("Sending OFFLINE status before disconnecting...")
        # payload_offline = {
        #     "truckId": TRUCK_ID,
        #     "status": "OFFLINE",
        #     "timestamp": time.time(),
        #     "location": {
        #         "latitude": last_clean_latitude, 
        #         "longitude": last_clean_longitude
        #     }
        # }
        # client.publish(topic_status, json.dumps(payload_offline), qos=1, retain=True)
        
        # # Wait a bit to ensure message is sent
        # time.sleep(0.5)
        save_last_location(last_clean_latitude, last_clean_longitude)
        client.loop_stop()
        # client.disconnect()
        sdr.stop_stream()
        f_csv.close() 
        gps_tracker.stop()
        


if __name__ == "__main__":
    main()