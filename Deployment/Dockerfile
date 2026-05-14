# 1. Immagine ufficiale NVIDIA per JetPack 6.2 (Orin Nano)
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# 2. Impostazioni per Python e per non far bloccare l'installazione
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 3. Installazione di GNU Radio, OSmoSDR, pip e tool di base
RUN apt-get update && apt-get install -y \
    gnuradio \
    gr-osmosdr \
    hackrf \
    libhackrf-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 4. Cartella di lavoro
WORKDIR /app

# 5. Copia e installa i pacchetti Python "standard"
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install "numpy<2.0.0" "pandas<2.2.0"
RUN pip3 install --ignore-installed sympy
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. MAGIA JETPACK 6: Installiamo ONNX Runtime GPU scaricandolo dal server ufficiale NVIDIA (Jetson AI Lab) per CUDA 12.6
RUN pip3 install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 7. Copia tutto il tuo progetto dentro il Docker
COPY . /app

# 8. Esecuzione
CMD ["python3", "main_optimized.py"]
