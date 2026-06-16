# N-MON: Portable GNSS Jamming Detection and Classification System

## Overview

N-MON is a portable GNSS monitoring platform designed to detect and classify radio-frequency jamming attacks in real time. The system combines Software Defined Radio (SDR) technology, signal processing, and Artificial Intelligence to provide continuous protection for GNSS-dependent assets such as logistics fleets, autonomous vehicles, critical infrastructure, and transportation systems.

Unlike traditional stationary monitoring solutions, N-MON is designed as a compact edge-computing device that can be installed directly on vehicles and operate autonomously without requiring constant connectivity or human supervision.

---

## Motivation

Modern society relies heavily on GNSS (GPS, Galileo, GLONASS, BeiDou) for navigation, synchronization, and asset tracking. Low-cost GNSS jammers are widely available and can disrupt these services, creating significant security risks.

A notable example is organized cargo theft, where attackers use jamming devices to disable fleet tracking systems and conceal vehicle movements.

N-MON was developed to provide an affordable and deployable monitoring solution capable of detecting these threats in real time and immediately notifying operators.

---

## System Architecture

The system consists of three main subsystems:

### Physical Device

* Multi-band GNSS Patch Antenna
* HackRF One SDR Front-End
* NVIDIA-based Embedded Processing Unit
* GNSS Receiver Module
* Wireless Communication Module (MQTT)

### Signal Processing Pipeline

1. GNSS RF signal acquisition using HackRF One
2. Complex I/Q sample capture at 10 MHz
3. Short-Time Fourier Transform (STFT)
4. Spectrogram generation
5. AI-based classification
6. Alert generation and transmission

### Control Center

* Real-time monitoring dashboard
* Driver notification interface
* Event history and filtering
* Fleet-wide monitoring

Communication between the edge device and the monitoring infrastructure is implemented using MQTT through a HiveMQ cloud broker.

---

## Signal Acquisition

### Hardware

* HackRF One SDR
* GNSS Patch Antenna
* Embedded NVIDIA Processing Unit

### Sampling Configuration

| Parameter         | Value                       |
| ----------------- | --------------------------- |
| Sampling Rate     | 10 MHz                      |
| Signal Type       | Complex I/Q                 |
| Processing Method | STFT Spectrogram            |
| Deployment        | Vehicle-mounted edge device |

The selected sampling rate allows coverage of the entire GPS L-band signal bandwidth while preserving sufficient frequency resolution for jamming analysis.

---

## AI-Based Jamming Classification

The project uses a lightweight deep-learning pipeline based on spectrogram analysis.

### Feature Extraction

Raw I/Q samples are converted into time-frequency representations using STFT. The resulting spectrograms are used as image-like inputs for the neural network.

### Classification Model

* ResNet18-based classifier
* Quantized INT8 inference for embedded deployment
* Real-time execution on edge hardware

### Supported Classes

The system detects and classifies:

* Clean GNSS Signals
* Linear Narrowband (LN)
* Linear Wide Fast Frequency Hopping (LWF)
* Ticking Jamming (TICK)
* Triangular Sweep Jamming (TRI)
* Triangular Wide Sweep Jamming (TRIW)

To reduce false alarms, an alert is generated only after three consecutive classifications of the same jamming type.

---

## Real-Time Alerting System

When a jammer is detected:

1. The edge device classifies the interference type.
2. An MQTT alert is published.
3. The Driver UI receives an immediate notification.
4. The Control Center dashboard records the event.
5. Vehicle position and event metadata are logged.

The communication layer supports:

* MQTT QoS Level 1 delivery
* Last Will and Testament (LWT)
* Automatic message recovery after reconnection
* Multi-vehicle topic separation

---

## User Interface

### Driver Interface

Provides:

* Current signal status
* Jamming alerts
* Device operational status

### Control Center Dashboard

Provides:

* Fleet monitoring
* Active vehicle tracking
* Jamming event history
* Event filtering
* Geographic visualization
* CSV export and restore functionality

Additional features:

* Clear History
* Restore History
* Persistent event logging

---

## Test Results

### Hardware & Acquisition

| Test                        | Result |
| --------------------------- | ------ |
| Continuous IQ Acquisition   | PASS   |
| Position + IQ Streaming     | PASS   |
| 1-Hour Autonomous Operation | PASS   |
| Startup Time                | 70 s   |

### AI Performance

| Metric                                  | Result                       |
| --------------------------------------- | ---------------------------- |
| Clean Signal Accuracy                   | 97%                          |
| Average Jamming Classification Accuracy | 95%                          |
| Low-JSR Robustness                      | 3/5 classes passed           |
| Unknown Jamming Detection               | 78% detected as interference |

### Communication

| Test                  | Result |
| --------------------- | ------ |
| Alert Delivery        | PASS   |
| LWT Offline Detection | PASS   |
| Network Recovery      | PASS   |
| Multi-Vehicle Routing | PASS   |

### User Interface

| Test                     | Result |
| ------------------------ | ------ |
| Driver UI Load Time      | < 5 s  |
| Control Center Load Time | < 10 s |
| Alert Visualization      | PASS   |
| Event Filtering          | PASS   |

---

## Performance Analysis

Current average end-to-end latency:

**350 ms**

Latency distribution:

| Stage                  | Contribution |
| ---------------------- | ------------ |
| Feature Extraction     | 51.6%        |
| Spectrogram Generation | 30.1%        |
| CNN Inference          | 9.2%         |
| INT8 Conversion        | 9.2%         |

The primary bottleneck is CPU-bound preprocessing rather than neural network inference.

---

## Dataset

The training dataset consists of:

### Clean Samples

Collected using multiple:

* Antennas
* Receiver configurations
* LNA setups

### Jammed Samples

Generated across multiple:

* Jamming waveforms
* Jamming-to-Signal Ratios (JSR)
* Power levels ranging from -5 dB to +30 dB

Due to the large volume of raw I/Q recordings (approximately 100 GB), the complete dataset is not included in this repository.

---

## Requirements Satisfaction

The final prototype successfully satisfies:

### User Requirements

* Plug-and-play deployment
* Real-time threat visualization
* Automatic alert generation

### Functional Requirements

* Continuous GNSS monitoring
* Time-frequency signal processing
* AI-based jamming classification
* Dashboard visualization
* Remote alert transmission

### Technical Requirements

* SDR-based RF acquisition
* Real-time AI inference
* STFT processing pipeline
* Synthetic jammer test environment
* MQTT communication infrastructure

---

## Future Improvements

### Open-Set Recognition

Current unknown-jammer detection:

* Current: 32.8%
* Target: >80%

Future work includes:

* Per-class threshold calibration
* Advanced Open-Set Recognition techniques
* Improved feature-space separation

### Performance Optimization

Current latency:

* Current: 350 ms
* Target: <100 ms

Potential improvements:

* ARM-optimized DSP libraries
* C++ preprocessing pipeline
* GPU-accelerated feature extraction

---

## Project Team

Group G — Interdisciplinary Project 2025/2026

* Lorenzo Braia
* Lorenzo Cavallaro
* Simone Peradotto

Politecnico di Torino

---

## Disclaimer

This repository contains software, documentation, and models developed for academic and research purposes. The generated dataset and some project assets are not publicly distributed due to storage requirements and licensing constraints.
