## Portable GNSS Jamming Detection and Classification ##
**Project Overview**
This project focuses on the development of a portable, edge-computing device capable of detecting and classifying GNSS (Global Navigation Satellite System) jamming attacks.

While standard detection systems are often stationary and bulky, this project aims to engineer a compact solution suitable for mobile deployment in vehicles (e.g., logistics trucks, cash-in-transit carriers, and autonomous fleets). The system interfaces with a HackRF One Software Defined Radio (SDR) to capture raw RF signals, processes them to extract time-frequency features, and leverages Machine Learning (CNN or GRU models) to classify interference patterns in real-time.

**System Architecture and Methodology**
The solution is designed to operate on an embedded hardware platform, moving away from desktop-based processing to a portable architecture. The data pipeline consists of the following stages:

- **Signal Acquisition:**

- Hardware: HackRF One (SDR) + Portable Single Board Computer (SBC).

- Sampling Rate: 10 MHz. This frequency is selected to cover the full bandwidth of the GPS signal, ensuring higher fidelity than standard 2 MHz narrowband captures.

- Data Format: Complex I/Q samples (In-phase and Quadrature) captured in 100-microsecond bursts.

- **Preprocessing:**

- Time-to-Frequency Conversion: The raw complex number array is transformed into the frequency domain using Fourier Transforms.

- Spectrogram Generation: The system generates a binary frequency matrix (128 frequency bins x ~1000 samples). This creates a visual representation (image) of the signal, where jamming patterns are more distinct than in the time domain.

- **Classification (Inference):**
- The generated spectrograms are fed into a Machine Learning model (Convolutional Neural Network or Light GRU).

- The model detects the presence of jamming and classifies the specific type of attack.

**Jamming Classifications**
The system is trained to identify and distinguish between the following specific interference waveforms:

- **Linear Narrowband Jamming:** A continuous wave signal that interferes with GNSS signals over a specific narrow frequency band.

- **Linear Wide Fast Frequency Hopping:** A jamming signal that rapidly changes its frequency over a wide band to evade static filters and detection.

- **Ticking Jamming:** A pulsed jamming signal that intermittently disrupts GNSS signals, characterized by a periodic "ticking" pattern in the time domain.

- **Triangular Wave Jamming:** A frequency-modulated jamming signal with a triangular waveform sweep.

**Objectives**
- **Monitor GNSS Signal Integrity:** continuously analyze the RF spectrum to detect anomalies and noise floor elevations associated with interference.

- **Classify Attack Vectors:** distinguish between different jamming types to provide actionable intelligence on the nature of the threat.

- **Hardware Optimization:** transition the existing desktop-based solution to a portable architecture, optimizing for storage (SSD), RAM, and processing power required to handle 19 kB/s throughput and 100 GB storage requirements.

- **Robustness:** ensure the model performs accurately across different hardware configurations, including setups with and without Low Noise Amplifiers (LNA) and varying antenna types.

**Applications**
- **Secure Logistics:** Protecting high-value asset transport vehicles from GPS-spoofing and jamming intended to mask routes or facilitate theft.

- **Aviation and Maritime:** Ensuring safe positioning and timing data in critical navigation zones.
- **Critical Infrastructure:** Securing systems that rely on GNSS for precise timing (e.g., telecommunications networks and power grids).

- **Autonomous Systems:** Providing a first layer of protection for self-driving vehicles against navigation attacks.

**Dataset Information**
The project utilizes a custom dataset generated to train the ML models. It includes:

- **Clean Signals:** Captured with various antenna configurations and LNA setups.

- **Jammed Signals:** Captured with interference powers ranging from -5 dB to 30 dB (Jamming-to-Signal ratio).
Note: Due to the high sampling rate and the large volume of raw binary I/Q data (approximated at 100 GB for the full training set), the dataset is not hosted directly in this repository.

**Context**
This work is part of the Navigation and Monitoring course studies, exploring the intersection of Radio Frequency (RF) Engineering, Digital Signal Processing, and Embedded Artificial Intelligence.