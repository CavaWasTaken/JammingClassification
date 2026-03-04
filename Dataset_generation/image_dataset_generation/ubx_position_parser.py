import serial
import serial.tools.list_ports
import time
from pyubx2 import UBXReader, UBX_PROTOCOL
import sys
import struct

UBX_SYNC = b'\xB5\x62'
ESF_STATUS_CLASS = 0x10
ESF_STATUS_ID = 0x10

# Custom logger function
def log_message(level, msg):
    entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}"
    print(entry)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(entry + "\n")

def find_ublox_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:

        description = port.description or ""
        manufacturer = port.manufacturer or ""

        if "u-blox" in description or "u-blox" in manufacturer:
            return port.device

        if (port.vid, port.pid) in [(0x1546, 0x01A7)]:  # Example for ZED-F9P
            return port.device
    return None

def fixType_meanings(fixType):
    return {
        0: "No Fix",
        1: "DR Only",
        2: "2D Fix",
        3: "3D Fix",
        4: "GNSS+DR",
        5: "Time Only Fix"
    }.get(fixType, "UNKNOWN")

def fusion_mode_meaning(mode):
    return {
        0: "Initialization",
        1: "Fusion",
        2: "Suspended Fusion",
        3: "Disabled Fusion"
    }.get(mode, "UNKNOWN")

def calib_status_meaning(status):
    return {
        0: "Not-Cal",
        1: "Cal-ING",
        2: "Cal-ED",
        3: "Cal-ED",
    }.get(status, "UNKNOWN")

def imu_status_meaning(status):
    return {
        0: "OFF",
        1: "INITIALIZING",
        2: "INITIALIZED"
    }.get(status, "UNKNOWN")

def sensor_type_name(sensor_type):
    return {
        14: "Gyro X",
        13: "Gyro Y",
        5: "Gyro Z",
        16: "Accel X",
        17: "Accel Y",
        18: "Accel Z",
    }.get(sensor_type, f"Unknown Type {sensor_type}")


def extract_latest_esf_status(data):

    latest_payload = None
    i = 0
    while i < len(data) - 8:
        if data[i:i + 2] == UBX_SYNC:
            msg_class = data[i + 2]
            msg_id = data[i + 3]
            length = struct.unpack_from("<H", data, i + 4)[0]
            payload_start = i + 6
            payload_end = payload_start + length

            if payload_end + 2 > len(data):  # Check for incomplete message
                break

            if msg_class == ESF_STATUS_CLASS and msg_id == ESF_STATUS_ID:
                latest_payload = data[payload_start:payload_end]

            i = payload_end + 2  # Skip to next message (+2 for checksum)
        else:
            i += 1

    if latest_payload:
        return parse_ubx_esf_status(latest_payload)
    else:
        return None


def parse_ubx_esf_status(payload):
    initStatus1 = payload[5]
    initStatus2 = payload[6]
    fusionMode = payload[12]
    numSens = payload[15]

    imuStatus = initStatus2 & 0x03  # bits 0–1
    insStatus = (initStatus1 >> 5) & 0x03  # bits 5–6

    fusion_status = f"Fusion Mode: {fusion_mode_meaning(fusionMode)} |  IMU: {imu_status_meaning(imuStatus)} |  INS: {imu_status_meaning(insStatus)} | Sensors: {numSens}"

    sensor_summaries = []
    for i in range(numSens):
        offset = 16 + i * 4
        if offset + 4 > len(payload):
            print(f"Sensor {i} out of bounds!")
            continue

        sensStatus1 = payload[offset]
        sensStatus2 = payload[offset + 1]

        sensor_type = sensStatus1 & 0x3F
        calibStatus = sensStatus2 & 0x03
        sensor_name = sensor_type_name(sensor_type)
        status_str = calib_status_meaning(calibStatus)
        sensor_summaries.append(f"{sensor_name}: {status_str}")

    sensor_summary_str = " | ".join(sensor_summaries)

    return fusion_status, sensor_summary_str

def parse_nav_pvt(payload):

    year = getattr(parsed, "year", None)
    month = getattr(parsed, "month", None)
    day = getattr(parsed, "day", None)
    hour = getattr(parsed, "hour", None)
    minute = getattr(parsed, "min", None)
    second = getattr(parsed, "second", None)

    fixType = getattr(parsed, "fixType", None)
    fixType_str = fixType_meanings(fixType)

    lat = getattr(parsed, "lat", None)
    lon = getattr(parsed, "lon", None)
    hMSL = getattr(parsed, "hMSL", None)* 1e-3

    hAcc = getattr(parsed, "hAcc", None) * 1e-3
    vAcc = getattr(parsed, "vAcc", None) * 1e-3
    return (
        f"{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}, "
        f"Fix: {fixType_str}, Lat: {lat:.4f}, Lon: {lon:.4f}, Hgt: {hMSL:.2f} m, "
        f"hAcc: {hAcc:.2f} m, vAcc: {vAcc:.2f} m")

# Configure the unified logger
log_file = "jamming_detection.log"

SERIAL_PORT = find_ublox_port()
if not SERIAL_PORT:
    print("u-blox receiver not found. Please connect the device.")
    sys.exit(1)

# Main Loop of code
try:
    # Open serial connection
    with serial.Serial(SERIAL_PORT, baudrate=230400, timeout=1) as ser:
        ubr = UBXReader(ser, protfilter=UBX_PROTOCOL)

        while True:

            start_time = time.time()
            while time.time() - start_time < 1:
                raw, parsed = ubr.read()
                if parsed is None:
                    continue

                if parsed.identity == "NAV-PVT":
                    pvt_result= parse_nav_pvt(parsed)
                    print(pvt_result if pvt_result else "Waiting for NAV-PVT message...")
                elif parsed.identity == "ESF-STATUS":
                    esf_result = extract_latest_esf_status(raw)
                    if esf_result:
                        esf_status, sensor_status = esf_result
                    else:
                        esf_status, sensor_status = None, None
                    print(esf_status if esf_status else "Waiting for ESF status...")
                    print(sensor_status if sensor_status else "Waiting for Sensor status...")

            time.sleep(0.5)

except KeyboardInterrupt:
    print("User interrupted logging.")
except Exception as e:
    log_message("ERROR", str(e))

