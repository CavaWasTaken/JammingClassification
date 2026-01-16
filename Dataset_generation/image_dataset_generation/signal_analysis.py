from io import BytesIO
import base64

import numpy as np
from scipy import signal as signal_sci
from scipy.stats import skew, kurtosis
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib
import os
# matplotlib.use('agg')
import math
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['savefig.pad_inches'] = 0


class SignalAnalysis:
    fs: int
    fc: int
    output_type: str
    file_name: str

    def __init__(self, fs=5000000, fc=0, output_type="sc8", file_name=None):
        # Parameter related to the file
        self.fs = fs  # Sampling Frequency
        self.fc = fc  # Carrier Frequency
        self.output_type = output_type
        self.file_name = file_name  # File_path for analysis
        self._memmap = None # memory-mapped file object
        self._use_fallback = False  # Flag to use file reading instead of memmap
        self._file_handle = None  # Persistent file handle for fallback

    def _get_memmap(self):
        """Try to get memmap, fallback to None if not supported"""
        if self._memmap is None and self.file_name is not None and not self._use_fallback:
            if self.output_type == "fc32":
                dtype = np.complex64
            elif self.output_type == "sc16":
                dtype = np.int16
            elif self.output_type == "sc8":
                dtype = np.int8
            else:
                raise ValueError(f"Unsupported output type: {self.output_type}")

            try:
                self._memmap = np.memmap(self.file_name, dtype=dtype, mode='r')
                print(f"Using memory-mapped file access for {os.path.basename(self.file_name)}")
            except (OSError, ValueError, IOError) as e:
                print(f"Warning: memmap not supported ({e})")
                print(f"Falling back to direct file reading (slower but compatible with Google Drive)")
                self._use_fallback = True
                self._memmap = None
        return self._memmap
    
    def _get_file_handle(self):
        """Get or create persistent file handle for fallback reading"""
        if self._file_handle is None and self.file_name is not None:
            self._file_handle = open(self.file_name, 'rb')
        return self._file_handle
    
    def close(self):
        """Close file handles. Call this when done processing a file."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        # Note: memmap doesn't need explicit closing, but we can clear the reference
        if self._memmap is not None:
            del self._memmap
            self._memmap = None

    def read_bin_file(self, window_interval, start_point=0):
        num_samples = int(window_interval * self.fs)
        
        # Try memmap first, fallback to file reading if needed
        memmap = self._get_memmap()
        
        if memmap is not None:
            # Fast path: use memory-mapped file
            if self.output_type == "fc32":
                offset_ = int(start_point * self.fs)    # offset in number of complex samples (not bytes)
                raw_signal = memmap[offset_: offset_ + num_samples].copy()  # read complex64 samples without loading entire file

            elif self.output_type == "sc16":
                offset_ = int(start_point * self.fs * 2)    # offset in number of int16 samples, *2 since each complex sample has 2 int16 components (I and Q)
                raw_signal = memmap[offset_: offset_ + num_samples * 2].copy()  # read int16 samples without loading entire file
                raw_signal = raw_signal.astype(np.float32).view(np.complex64)   # convert to complex64

            elif self.output_type == "sc8":
                offset_ = int(start_point * self.fs * 2)    # offset in number of int8 samples, *2 since each complex sample has 2 int8 components (I and Q)
                raw_signal = memmap[offset_: offset_ + num_samples * 2].copy()  # read int8 samples without loading entire file
                raw_signal = raw_signal.astype(np.float32).view(np.complex64)   # convert to complex64
        else:
            # Fallback path: use file handle with seek (compatible with Google Drive)
            fh = self._get_file_handle()
            
            if self.output_type == "fc32":
                bin_type = np.complex64
                offset_bytes = int(start_point * self.fs * 8)  # 8 bytes per complex64
                count = num_samples
                
            elif self.output_type == "sc16":
                bin_type = np.int16
                offset_bytes = int(start_point * self.fs * 4)  # 4 bytes per complex sample (2 int16s)
                count = num_samples * 2
                
            elif self.output_type == "sc8":
                bin_type = np.int8
                offset_bytes = int(start_point * self.fs * 2)  # 2 bytes per complex sample (2 int8s)
                count = num_samples * 2
            
            # Seek to position and read
            fh.seek(offset_bytes)
            raw_signal = np.fromfile(fh, dtype=bin_type, count=count)
            
            # Convert to complex64 for sc16 and sc8
            if self.output_type in ["sc16", "sc8"]:
                raw_signal = raw_signal.astype(np.float32).view(np.complex64)

        return raw_signal

    def spectrogram_image(self, start_point=0, window_interval=100e-6, nfft=128, window=None, overlap_percentage=0.999):

        if self.file_name is not None:
            raw_signal_ = self.read_bin_file(window_interval, start_point)
            # print(raw_signal_.shape)
            if overlap_percentage is not None:
                number_overlap = math.floor(nfft * overlap_percentage)
            else:
                number_overlap = None

            if window is not None:
                if window == 'kaiser':
                    window = ('kaiser', 5.0)
                win = signal_sci.get_window(window, nfft)
            else:
                win = signal_sci.get_window(('kaiser', 5.0), nfft)

            spec, freq, t = mlab.specgram(x=raw_signal_,
                                          Fs=self.fs,
                                          NFFT=nfft,
                                          window=win,
                                          noverlap=number_overlap)

            # freq += self.fc
            # matplotlib.pyplot.close()
            # plt.clf()
            # fig = plt.figure()
            # plt.pcolormesh(t / 1e-6, freq / 1e6, spec, shading='auto')
            # plt.axis('off')
            #fig.axes.get_xaxis().set_visible(False)
            #fig.axes.get_yaxis().set_visible(False)

            # ax = plt.gca()
            # b_ = self.fs / 2
            # ax.set_ylim([(-b_ / 2 + self.fc) / 1e6, (b_ / 2 + self.fc) / 1e6])
            #plt.ylabel("Frequency " + r"$[MHz]$")
            #plt.xlabel("Time " + r"$[\mu \, s]$")
            #plt.title("Spectrogram")
            # plt.show()
            return spec

    def spectrum_image(self, start_point=0, window_interval=100e-6, nfft=128, window=None, overlap_percentage=None):
        if self.file_name is not None:
            raw_signal_ = self.read_bin_file(window_interval, start_point)

            if overlap_percentage is not None:
                number_overlap = math.floor(nfft * overlap_percentage)
            else:
                number_overlap = None

            if window is not None:
                if window == 'kaiser':
                    window = ('kaiser', 5.0)
                win = signal_sci.get_window(window, nfft)
            else:
                win = signal_sci.get_window(('kaiser', 5.0), nfft)

            # First Method to Calculate PSD
            # freq, psd = signal_sci.welch(raw_signal_,
            #                              fs=self.fs,
            #                              nfft=nfft,
            #                              window=win,
            #                              noverlap=number_overlap,
            #                              scaling='density')
            #
            # freq = np.fft.fftshift(freq)
            # psd = np.fft.fftshift(psd)

            # Second Method to Calculate PSD (based SONG )
            psd, freq = mlab.psd(raw_signal_,
                                 Fs=self.fs,
                                 NFFT=nfft,
                                 window=win,
                                 sides='twosided',
                                 noverlap=number_overlap, scale_by_freq=True)

            psd = (10. * np.log10(psd) + 300) - 300
            freq += self.fc

            plt.close('all')
            plt.clf()
            fig = plt.figure()
            plt.plot(freq / 1e6, psd)
            plt.xlabel("Frequency " + r"$[MHz]$")
            plt.ylabel("PSD" + r"$[dBW/Hz]$")
            plt.title("Spectrum")
            return fig

    def convert_fig_to_byte(self, fig, dpi=None):
        image = BytesIO()
        fig.savefig(image, format='png', dpi=dpi)
        image_bin = base64.b64encode(image.getvalue()).decode("utf-8")
        return image_bin

    def convert_fig_to_numpy(self, spectrogram_img):
        spectrogram_img.axes[0].get_xaxis().set_visible(False)
        spectrogram_img.axes[0].get_yaxis().set_visible(False)
        spectrogram_img.axes[0].title.set_visible(False)
        spectrogram_img.tight_layout(pad=0, h_pad=0, w_pad=0)
        spectrogram_img.canvas.draw()
        buf = spectrogram_img.canvas.tostring_rgb()
        ncols, nrows = spectrogram_img.canvas.get_width_height()
        image_array = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        return image_array

    def get_input_detection(self, start_point=0, window_interval=200e-6, len_seq=4, nfft=256):
        raw_signal_ = self.read_bin_file(window_interval * len_seq, start_point)
        step_size = int(window_interval * self.fs)
        after_fft = []
        for i in range(0, len_seq):
            after_fft.append(self.do_fft(raw_signal_[i * step_size: (i + 1) * step_size], nfft, self.fs))
        after_fft = np.array(after_fft)
        return after_fft

    # def do_fft(self, col, window_len, BW=25e6):
    #     """ Calculates FFT for the 1 ms snapshot. Rearranges the resulted in
    #         tensor with FFTshift.
    #         In addition, the resulted in tensor is scaled as follows:
    #         Real unit is scaled with maximum absolute value of real units
    #         within the tensor.
    #         Imaginary unit is scaled with maximum absolute value of imaginary units
    #         within the tensor.
    #     Args:
    #         df (pd.DataFrame): Dataframe from which the snapshot is taken
    #         n (Integer): Window length for FFT
    #         packsize (Integer): Number of observations in one snapshot.
    #         key (String): Column name,
    #         df_to_add (pd.DataFrame): Dataframe where the resulting in vector
    #         plot_ftt (bool, optional): Tells whether to plot the given results. Defaults to False.
    #     """
    #     nblock = window_len
    #     overlap_percentage = 0.5
    #     number_overlap = math.floor(nblock * overlap_percentage)
    #     win = signal_sci.get_window('hann', nblock)
    #     _, pxx = signal_sci.welch(col, BW, window=win, noverlap=number_overlap, nfft=nblock, return_onesided=False)
    #     input_val = torch.fft.fftshift(10 * torch.log10(torch.tensor(pxx)))
    #
    #     data = input_val.numpy().astype(np.float64) / window_len
    #     # input_val = 10. * np.log10(np.fft.fftshift(pxx))
    #
    #     # scaler = MinMaxScaler(feature_range=(-1, 1))
    #     # data = scaler.fit_transform(input_val.reshape(-1, 1)).flatten()
    #     return data

    def extract_features(self, start_point=0, window_interval=200e-6):
        raw_signal_ = self.read_bin_file(window_interval, start_point)
        real_raw_signal_ = raw_signal_.real

        mean_ = np.mean(real_raw_signal_)
        median_ = np.median(real_raw_signal_)
        std_ = np.std(real_raw_signal_)
        mad_ = np.mean(np.absolute(real_raw_signal_ - np.mean(real_raw_signal_, axis=None)), axis=None)
        rms_ = np.sqrt(np.mean(real_raw_signal_ ** 2))
        percentile_25th_ = np.quantile(real_raw_signal_, 0.25)
        percentile_75th_ = np.quantile(real_raw_signal_, 0.75)
        iqr_ = np.subtract(*np.percentile(real_raw_signal_, [75, 25]))
        skewness_ = skew(real_raw_signal_)
        kurtosis_ = kurtosis(real_raw_signal_, fisher=False, bias=True)
        entropy_ = self.signal_entropy(real_raw_signal_)
        max_power_win_, freq_max_power_, mean_power_win_ = self.frequency_features(raw_signal_)

        pentropy = self.pentropy(real_raw_signal_)
        pentropy_mean_ = np.mean(pentropy)
        pentropy_std_ = np.std(pentropy)

        features_list = [mean_, median_, std_, mad_, rms_, percentile_25th_, percentile_75th_,
                         iqr_, skewness_, kurtosis_, entropy_, max_power_win_, freq_max_power_,
                         mean_power_win_, pentropy_mean_, pentropy_std_]
        return features_list

    def frequency_features(self, raw_signal_):
        nfft = 512
        win = signal_sci.get_window('boxcar', nfft)
        raw_signal_ = raw_signal_ - np.mean(raw_signal_)
        freq, psd = signal_sci.welch(raw_signal_,
                                     fs=self.fs,
                                     nfft=nfft,
                                     window=win,
                                     noverlap=0, return_onesided=False)

        maxpower_win_ = np.max(psd)
        numb = np.where(psd == maxpower_win_)[0]
        freq_maxpower_ = freq[numb].tolist()[0]
        meanpower_win_ = np.mean(psd)
        return freq_maxpower_, maxpower_win_, meanpower_win_

    def signal_entropy(self, sig):
        h, descriptor = self.histogram_signalEntropy(sig, descriptor=None)

        lowerbound, upperbound, ncell = descriptor

        estimate, sigma, count = 0, 0, 0

        for n in range(ncell):
            if h[n] != 0:
                logf = np.log(h[n])
            else:
                logf = 0
            count += h[n]
            estimate -= h[n] * logf
            sigma += h[n] * logf ** 2

        estimate = estimate / count
        sigma = np.sqrt((sigma / count - estimate ** 2) / (count - 1))
        estimate = estimate + np.log(count) + np.log((upperbound - lowerbound) / ncell)
        nbias = -(ncell - 1) / (2 * count)

        # unbiased estimate
        estimate = estimate - nbias
        nbias = 0

        base = np.e
        estimate = estimate / np.log(base)
        nbias = nbias / np.log(base)
        sigma = sigma / np.log(base)

        return estimate

    def histogram_signalEntropy(self, x, descriptor=None):
        if x.ndim != 1:
            raise ValueError("Invalid dimension of x")

        NColX = len(x)

        if descriptor is None:
            minx = np.min(x)
            maxx = np.max(x)
            delta = (maxx - minx) / (len(x) - 1)
            ncell = int(np.ceil(np.sqrt(len(x))))
            descriptor = [minx - delta / 2, maxx + delta / 2, ncell]

        lower, upper, ncell = descriptor

        if ncell < 1:
            raise ValueError("Invalid number of cells")

        if upper <= lower:
            raise ValueError("Invalid bounds")

        result = np.zeros(ncell)
        y = np.round((x - lower) / (upper - lower) * ncell + 0.5)

        for n in range(NColX):
            index = int(y[n])
            if 1 <= index <= ncell:
                result[index - 1] += 1

        return result, descriptor

    def pentropy(self, real_raw_signal_):
        nfft = 1024
        time_frame = 129
        overlap = (len(real_raw_signal_) - nfft) / (time_frame - 1)
        noverlap = np.floor(overlap)
        win = signal_sci.get_window(('kaiser', 20.0), 1024)

        _, _, Sxx = signal_sci.spectrogram((real_raw_signal_ / np.sum(win)) * math.sqrt(2), self.fs,
                                           nfft=2048,
                                           nperseg=1024,
                                           return_onesided=True,
                                           noverlap=993, scaling='spectrum')
        normalized_spectrum = Sxx / np.sum(Sxx, axis=0)  # Normalize the power spectrogram
        entropy = np.sum(-normalized_spectrum * np.log2(normalized_spectrum), axis=0)
        entropy /= np.log2(normalized_spectrum.shape[0])
        return entropy
