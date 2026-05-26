import streamlit as st
import paho.mqtt.client as mqtt
import pandas as pd
from datetime import datetime
import json
import time
from queue import Queue, Empty
import ssl
import pydeck as pdk
import os

#digli si ignorare i messaggi di status = None
# Page config
st.set_page_config(page_title="N-MON Control Center", layout="wide")


DEDUP_WINDOW_SECONDS = 120
VEHICLE_TIMEOUT_SECONDS = 300

CSV_FILENAME = "alerts_history.csv"

def save_history_to_csv():
    """Save the current table to a CSV in append mode (add rows)."""
    if not st.session_state.alerts:
        return
        
    df = pd.DataFrame(st.session_state.alerts)
    # Check if the file already exists to decide whether to write column headers
    file_exists = os.path.isfile(CSV_FILENAME)
    
    # mode='a' means 'append'. If the file isn't there, it creates it.
    df.to_csv(CSV_FILENAME, mode='a', index=False, header=not file_exists)

def restore_history_from_csv():
    """Load the CSV, sort by date and repopulate the session state."""
    if not os.path.isfile(CSV_FILENAME):
        st.warning("No history file found!")
        return
        
    df = pd.read_csv(CSV_FILENAME)
    if df.empty:
        return
        
    # Let's sort from newest to oldest (as expected by the app)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp_dt', ascending=False).drop(columns=['timestamp_dt'])
    
    # We remove any total duplicates created by incorrect multiple saves
    df = df.drop_duplicates()
    
    # Let's repopulate the alerts
    st.session_state.alerts = df.to_dict('records')
    
    # Let's rebuild the deduplication index (alert_last_seen)
    st.session_state.alert_last_seen = {}
    for idx, alert in enumerate(st.session_state.alerts):
        key = (alert["vehicle_id"], alert["type"], alert["level"])
        if key not in st.session_state.alert_last_seen:
            st.session_state.alert_last_seen[key] = idx


# --- GLOBAL QUEUES ---
@st.cache_resource
def get_alert_queue():
    return Queue()

@st.cache_resource
def get_status_queue():
    return Queue()

alert_queue = get_alert_queue()
status_queue = get_status_queue()

@st.cache_resource
def get_system_state():
    return {"mqtt_connected": False}

system_state = get_system_state()

# --- Streamlit Session State ---
if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "active_vehicles" not in st.session_state:
    st.session_state.active_vehicles = {}

if "show_filters" not in st.session_state:
    st.session_state.show_filters = False

if "show_map" not in st.session_state:
    st.session_state.show_map = False

if "filter_vehicle_id" not in st.session_state:
    st.session_state.filter_vehicle_id = []
if "filter_type" not in st.session_state:
    st.session_state.filter_type = []
if "filter_level" not in st.session_state:
    st.session_state.filter_level = []
if "filter_time_start" not in st.session_state:
    st.session_state.filter_time_start = ""
if "filter_time_end" not in st.session_state:
    st.session_state.filter_time_end = ""

# ------------------------------------------------------------------
# DEDUPLICATION HELPER
# alert_last_seen maps (vehicle_id, type, level) → index in st.session_state.alerts
# where index 0 = most recent (list is kept newest-first).
# ------------------------------------------------------------------
if "alert_last_seen" not in st.session_state:
    # key: (vehicle_id, type, level)  →  value: index inside st.session_state.alerts
    st.session_state.alert_last_seen = {}

def upsert_alert(new_alert: dict) -> bool:
    """
    Insert or update an alert applying the deduplication window.

    Returns True if a NEW row was created, False if an existing row was
    updated in-place (so the caller knows whether to increment the counter).
    """
    # CHIAVE DI DEDUPLICAZIONE AGGIORNATA
    key = (new_alert["vehicle_id"], new_alert["type"], new_alert["level"])
    new_ts = datetime.strptime(new_alert["timestamp"], "%Y-%m-%d %H:%M:%S")

    existing_idx = st.session_state.alert_last_seen.get(key)

    if existing_idx is not None:
        # Make sure the index is still valid (e.g. after a Clear History)
        if existing_idx < len(st.session_state.alerts):
            existing_alert = st.session_state.alerts[existing_idx]
            existing_ts = datetime.strptime(existing_alert["timestamp"], "%Y-%m-%d %H:%M:%S")
            delta = abs((new_ts - existing_ts).total_seconds())

            if delta <= DEDUP_WINDOW_SECONDS:
                # ---- SAME EVENT: update position + timestamp in-place ----
                st.session_state.alerts[existing_idx] = new_alert
                # The row stays at the same position in the list (no reorder),
                # but the displayed data reflects the latest GPS fix.
                return False   # not a new row

    # ---- NEW EVENT: prepend to the list ----
    st.session_state.alerts.insert(0, new_alert)

    # After inserting at position 0, every previous index shifts by +1.
    # Update all stored indices accordingly.
    updated_last_seen = {}
    for k, idx in st.session_state.alert_last_seen.items():
        updated_last_seen[k] = idx + 1
    st.session_state.alert_last_seen = updated_last_seen

    # Store the index of the newly inserted alert (always 0)
    st.session_state.alert_last_seen[key] = 0

    return True   # new row created


# ---------------- MQTT CALLBACK ----------------
def on_message(client, userdata, msg):
    print(f"Messaggio ricevuto su {msg.topic}")
    with open("config.json") as f:
        cfg = json.load(f)
    topic_alert = cfg["topics"]["alerts"]
    topic_status = cfg["topics"]["status"]

    payload = json.loads(msg.payload.decode("utf-8", errors="ignore"))
    # If the payload does not have a timestamp (e.g. simplified LWT), use the current time
    if 'timestamp' in payload:
        ts = datetime.fromtimestamp(payload['timestamp'])
    else:
        ts = datetime.now()
    vehicle_id = msg.topic.split("/")[-1]

    if msg.topic.startswith(f"{topic_alert}/"):
        loc = payload.get('location', {})
        alert = {
            "vehicle_id": vehicle_id,
            "type": payload['type'],
            "level" : payload['level'],
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")
        }
        alert_queue.put(alert)

    elif msg.topic.startswith(f"{topic_status}/"):
        payload_status = payload.get('status')
        loc = payload.get('location', {})

        if payload_status is None:
            payload_status = "OFFLINE"
        
        # loc = payload.get('location', {})
        # status = payload.get('status', 'UNKNOWN')
        status = {
            "vehicle_id": vehicle_id,
            "status": payload_status,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
        }
        status_queue.put(status)

# --------------- MQTT CONNECTION ----------------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        system_state["mqtt_connected"] = True
        print("MQTT connected successfully")

def on_disconnect(client, userdata, rc):
    system_state["mqtt_connected"] = False
    print("MQTT disconnected")

@st.cache_resource
def init_mqtt():
    try:
        client = mqtt.Client("Control Centre", clean_session=False)
        client.on_message = on_message

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect

        with open("config.json") as f:
            cfg = json.load(f)

        host = cfg["hivemq"]["host"]
        port = cfg["hivemq"]["port_tcp"]
        user = cfg["hivemq"]["username"]
        pwd = cfg["hivemq"]["password"]
        topic_alert = cfg["topics"]["alerts"]
        topic_status = cfg["topics"]["status"]

        client.tls_set(cert_reqs=ssl.CERT_NONE)
        client.username_pw_set(user, pwd)
        client.connect(host, port)
        client.subscribe(f"{topic_alert}/#")
        client.subscribe(f"{topic_status}/#")
        client.loop_start()
        print("MQTT connected")
        return client
    except Exception as e:
        print(f"Errore MQTT: {e}")
        system_state["mqtt_connected"] = False
        return None

client = init_mqtt()

# ----------- FILTER LOGIC ----------------
def get_filtered_df():
    if not st.session_state.alerts:
        return pd.DataFrame()

    df = pd.DataFrame(st.session_state.alerts)

    if st.session_state.filter_vehicle_id:
        df = df[df["vehicle_id"].isin(st.session_state.filter_vehicle_id)]

    if st.session_state.filter_type:
        df = df[df["type"].isin(st.session_state.filter_type)]

    # LOGICA AGGIUNTA: Filtro per Jamming Level
    if st.session_state.filter_level:
        df = df[df["level"].isin(st.session_state.filter_level)]

    if st.session_state.filter_time_start:
        try:
            ts = pd.to_datetime(st.session_state.filter_time_start)
            df = df[pd.to_datetime(df["timestamp"]) >= ts]
        except Exception:
            pass

    if st.session_state.filter_time_end:
        try:
            te = pd.to_datetime(st.session_state.filter_time_end)
            df = df[pd.to_datetime(df["timestamp"]) <= te]
        except Exception:
            pass

    return df


# ===========================================================
@st.fragment(run_every=1)
def live_dashboard():
    # ----------------------------------------------------------
    # Process alert queue with deduplication
    # ----------------------------------------------------------
    new_alerts_count = 0   # counts only genuinely NEW rows
    while not alert_queue.empty():
        try:
            item = alert_queue.get_nowait()
            is_new = upsert_alert(item)
            if is_new:
                new_alerts_count += 1
        except Empty:
            break

    current_vehicle_count = len(st.session_state.active_vehicles)
    while not status_queue.empty():
        try:
            item = status_queue.get_nowait()
            vid = item["vehicle_id"]
            
            # If the incoming message (e.g. the LWT) has no coordinates or are None
            if not item.get("latitude") and not item.get("longitude"):
                # Let's check if we already knew the last location of this vehicle
                if vid in st.session_state.active_vehicles:
                    # copy the last known coordinates into the new message
                    item["latitude"] = st.session_state.active_vehicles[vid].get("latitude")
                    item["longitude"] = st.session_state.active_vehicles[vid].get("longitude")
                else:
                    if item["status"] == "OFFLINE":
                        continue
                    

            st.session_state.active_vehicles[vid] = item
        except Empty:
            break

    #removing inactive vehicles based on timeout
    current_time = datetime.now()
    vids_to_remove = []
    
    for vid, v_data in st.session_state.active_vehicles.items():
        last_seen = datetime.strptime(v_data["timestamp"], "%Y-%m-%d %H:%M:%S")
        
        if (current_time - last_seen).total_seconds() > VEHICLE_TIMEOUT_SECONDS:
            vids_to_remove.append(vid)
            
    for vid in vids_to_remove:
        del st.session_state.active_vehicles[vid]



    new_vehicles_delta = len(st.session_state.active_vehicles) - current_vehicle_count

    if new_alerts_count > 0:
        st.toast(f"{new_alerts_count} new jamming event(s) detected!", icon="🚨")

    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Vehicles", len(st.session_state.active_vehicles),
                f"+{new_vehicles_delta}" if new_vehicles_delta > 0 else "")
    col2.metric("Jamming Events", len(st.session_state.alerts),
                f"+{new_alerts_count}" if new_alerts_count > 0 else "")
    # NUOVO: Aggiornamento dinamico dello stato del sistema
    if system_state["mqtt_connected"]:
        col3.metric("System Status", "OPERATIONAL", "CONNECTED" ,delta_color="normal" )
    else:
        col3.metric("System Status", "OFFLINE", "DISCONNECTED" , delta_color="inverse")

    st.divider()
    st.subheader("Live Alerts Feed")

    # --- Buttons ---
    btn_col1, btn_col2, btn_col3, btn_col4, _ = st.columns([1.5, 1, 1, 1.5, 6] , gap="small")

    if btn_col1.button("Clear History" , use_container_width=True):
        save_history_to_csv()

        st.session_state.alerts = []
        st.session_state.alert_last_seen = {}   # reset dedup index too
        st.session_state.filter_vehicle_id = []
        st.session_state.filter_type = []
        st.session_state.filter_level = [] # Reset del nuovo filtro
        st.session_state.filter_time_start = ""
        st.session_state.filter_time_end = ""
        st.rerun(scope="fragment")

    if btn_col4.button("Restore History" , use_container_width=True):
        restore_history_from_csv()
        st.rerun(scope="fragment")

    if btn_col2.button("Filters", type="secondary" , use_container_width=True):
        st.session_state.show_filters = not st.session_state.show_filters

    if btn_col3.button("Map", type="secondary", use_container_width=True):
        st.session_state.show_map = not st.session_state.show_map

    # --- Filter panel ---
    if st.session_state.show_filters:
        if st.session_state.alerts:
            df_all = pd.DataFrame(st.session_state.alerts)
            with st.expander("Filter Options", expanded=True):
                # Cambiato in 4 colonne per far spazio al Level in fcol3
                fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                with fcol1:
                    st.session_state.filter_vehicle_id = st.multiselect(
                        "Vehicle ID",
                        options=sorted(df_all["vehicle_id"].unique().tolist()),
                        default=st.session_state.filter_vehicle_id
                    )
                with fcol2:
                    st.session_state.filter_type = st.multiselect(
                        "Jamming Type",
                        options=sorted(df_all["type"].unique().tolist()),
                        default=st.session_state.filter_type
                    )
                with fcol3:
                    # Inserito in mezzo come richiesto
                    st.session_state.filter_level = st.multiselect(
                        "Jamming Level",
                        options=sorted(df_all["level"].unique().tolist()),
                        default=st.session_state.filter_level
                    )
                with fcol4:
                    st.session_state.filter_time_start = st.text_input(
                        "Time From (YYYY-MM-DD HH:MM:SS)",
                        value=st.session_state.filter_time_start,
                        placeholder="e.g. 2025-01-01 08:00:00"
                    )
                    st.session_state.filter_time_end = st.text_input(
                        "Time To (YYYY-MM-DD HH:MM:SS)",
                        value=st.session_state.filter_time_end,
                        placeholder="e.g. 2025-12-31 23:59:59"
                    )
        else:
            st.info("No alerts to filter yet.")

    # --- Alerts table ---
    df_filtered = get_filtered_df()

    if not df_filtered.empty:
        st.dataframe(
            df_filtered,
            width='stretch',
            column_config={
                "vehicle_id": "Vehicle ID",
                "type": "Jamming Type",
                "level": "Jamming Level",
                "latitude": st.column_config.NumberColumn("Latitude", format="%.6f"),
                "longitude": st.column_config.NumberColumn("Longitude", format="%.6f"),
                "timestamp": "Time",
            },
            column_order=["vehicle_id", "timestamp", "type", "level", "latitude", "longitude"]
        )
    elif st.session_state.alerts:
        st.warning("No alerts match the current filters.")
    else:
        st.info("No alerts available. Listening...")

    # --- Map section ---
    if st.session_state.show_map:
        st.divider()
        st.subheader("Alert Map")

        source_df = df_filtered if not df_filtered.empty else pd.DataFrame(st.session_state.alerts)

        map_rows = []
        for _, row in source_df.iterrows():
            try:
                lat = row.get("latitude")
                lon = row.get("longitude")
                if lat is not None and lon is not None:
                    map_rows.append({
                        "lat": float(lat),
                        "lon": float(lon),
                        "vehicle_id": row["vehicle_id"],
                        "level": row["level"],
                        "type": row["type"],
                        "timestamp": row["timestamp"]
                    })
            except Exception as e:
                print(f"Map row error: {e}")

        if map_rows:
            map_df = pd.DataFrame(map_rows)
            filters_active = (
                bool(st.session_state.filter_vehicle_id) or
                bool(st.session_state.filter_type) or
                bool(st.session_state.filter_level) or # Verifica sul nuovo filtro
                bool(st.session_state.filter_time_start) or
                bool(st.session_state.filter_time_end)
            )
            st.caption(f"Showing {len(map_df)} alert location(s). Filters: {'Active ✅' if filters_active else 'None'}")
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_fill_color='[255, 60, 60, 200]',
                get_radius=70,
                radius_min_pixels=1,
                pickable=True,
            )

            view = pdk.ViewState(
                latitude=map_df["lat"].mean(),
                longitude=map_df["lon"].mean(),
                zoom=11,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view,
                    map_style="road",
                    tooltip={
                        "text": "Plate: {vehicle_id}\n Jamming: {type}-{level}\n Time: {timestamp}"
                    }
                )
            )
        else:
            st.info(
                "📍 No alert location data available."
            )

    # --- Active Vehicles ---
    with st.expander("See Active Vehicles"):
        if st.session_state.active_vehicles:
            st.dataframe(
                pd.DataFrame(list(st.session_state.active_vehicles.values())),
                width='stretch',
                column_config={
                    "vehicle_id": "Vehicle ID",
                    "status": "Status",
                    "timestamp": "Last Seen",
                    "latitude": st.column_config.NumberColumn("Latitude", format="%.6f"),
                    "longitude": st.column_config.NumberColumn("Longitude", format="%.6f"),
                },
                column_order=["vehicle_id", "status", "timestamp", "latitude", "longitude"]
            )
        else:
            st.write("No active vehicles.")


# ----------- RENDER ----------------
st.markdown("<h1 style='text-align: center;'>N-MON Fleet Monitor</h1>", unsafe_allow_html=True)
live_dashboard()