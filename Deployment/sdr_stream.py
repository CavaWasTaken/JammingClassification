import numpy as np
import collections
import threading
from sdr_base import SdrBase

class SdrStream(SdrBase):
    def __init__(self, *args, buffer_size=500e-3, buffer_length=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_buffer_size(buffer_size)

        self.iq_buffer = collections.deque(maxlen=buffer_length)  # Circular buffer
        self.stream_thread = None
        self.stop_flag = threading.Event()
        self.iq = None

    def run_background_stream(self):
        """
        Run the SDR flowgraph in a separate thread, continuously acquiring data.
        """
        self.start()
        self.flowgraph_started.set()
        print(f'--- SDR streaming thread started.')

        while not self.stop_flag.is_set():
            if len(self.blocks_vector_sink_x_0.data()) > 0.5 * self.num_samples_buffer:
                iq_data = self.blocks_vector_sink_x_0.data()
                self.iq_buffer.append(iq_data)  # Store in buffer

                # self.blocks_head_0_0.reset()
                self.blocks_vector_sink_x_0.reset()
            else:
                self.stop_flag.wait(0.001)  # Efficient waiting without CPU overuse

        print(f'--- SDR streaming thread finished.')
        self.stop()
        self.wait()

    def start_stream(self):
        """
        Starts the background SDR stream.
        """
        if self.stream_thread is None or not self.stream_thread.is_alive():
            self.stop_flag.clear()
            self.stream_thread = threading.Thread(target=self.run_background_stream, daemon=True)
            self.stream_thread.start()

    def stop_stream(self):
        """
        Stops the background SDR stream.
        """
        self.stop_flag.set()
        self.flowgraph_started.clear()  # Ensure it's reset
        self.iq_buffer.clear()
        self.stream_thread= None


    def get_samples(self, window_chunk=200e-6):
        """
        Fetches the latest `num_samples` from the most recent IQ data in the buffer.
        """
        if not self.iq_buffer:
            print("--- No IQ data available!")
            return None

        num_samples = int(window_chunk * self.samp_rate)

        # vector_sink_c already provides complex IQ samples.
        iq_data = np.asarray(self.iq_buffer[-1], dtype=np.complex64)[-num_samples:]

        if np.array_equal(iq_data, self.iq):  # True
            print(f'--- IQ data of two streaming are equal.')
        else:
            max_length = max(map(len, self.iq_buffer))  # Cleaner & faster!
            print(f"--- Maximum IQ buffer element length: {max_length}")

        self.iq = iq_data
        return iq_data


    def sig_handler(self):
        self.stop_flag.set()
        self.iq_buffer.clear()
        self.stream_thread = None
        return True

