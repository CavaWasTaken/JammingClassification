import numpy as np
from scipy import signal as signal_sci
from scipy.stats import skew, kurtosis
import matplotlib.mlab as mlab
# matplotlib.use('agg')
import math
import torch

class SignalAnalysisLive:
    fs: int

    def __init__(self, fs=5000000):
        # Parameter related to the file
        self.fs = fs  # Sampling Frequency
        self._use_fallback = False  # Flag to use file reading instead of memmap
        self._file_handle = None  # Persistent file handle for fallback

    def compute_spectrogram(self, raw_signal_100us, nfft=128, window=None, overlap_percentage=0.999):
        """
        Compute spectrogram using same parameters as signal_analysis.spectrogram_image()
        Returns spectrogram array (nfft, time_steps)
        """

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
    
        spec, freq, t = mlab.specgram(x=raw_signal_100us,
                                    Fs=self.fs,
                                    NFFT=nfft,
                                    window=win,
                                    noverlap=number_overlap)
        return spec, freq, t
    
    def compute_spectrogram_general(self, raw_signal_100us, nfft=128, window=None, overlap_percentage=0.999): 
        """
        Universal Spectrogram: Use GPU if available, CPU otherwise.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if overlap_percentage is not None:
            number_overlap = math.floor(nfft * overlap_percentage)
        else:
            number_overlap = 0
            
        hop_length = nfft - number_overlap

        signal_tensor = torch.tensor(raw_signal_100us, dtype=torch.float32, device=device)

        if window == 'kaiser' or window is None:
            win_np = signal_sci.get_window(('kaiser', 5.0), nfft)
        else:
            win_np = signal_sci.get_window(window, nfft)
            
        win_tensor = torch.tensor(win_np, dtype=torch.float32, device=device)

        stft_result = torch.stft(
            signal_tensor, 
            n_fft=nfft, 
            hop_length=hop_length, 
            win_length=nfft, 
            window=win_tensor,
            center=False,
            return_complex=True
        )
        
        spec = torch.abs(stft_result).cpu().numpy()
        
        return spec, None, None
    
    def extract_features_direct(self, raw_signal_200us):
        """
        Extract 16 features directly from in-memory signal.
        Uses same logic as signal_analysis.extract_features() but works on raw signal.
        """

        real_raw_signal = raw_signal_200us.real if np.iscomplexobj(raw_signal_200us) else raw_signal_200us
        
        mean_ = np.mean(real_raw_signal)
        median_ = np.median(real_raw_signal)
        std_ = np.std(real_raw_signal)
        mad_ = np.mean(np.absolute(real_raw_signal - np.mean(real_raw_signal)))
        rms_ = np.sqrt(np.mean(real_raw_signal ** 2))
        percentile_25th_ = np.quantile(real_raw_signal, 0.25)
        percentile_75th_ = np.quantile(real_raw_signal, 0.75)
        iqr_ = np.subtract(*np.percentile(real_raw_signal, [75, 25]))
        skewness_ = skew(real_raw_signal)
        kurtosis_ = kurtosis(real_raw_signal, fisher=False, bias=True)
        entropy_ = self.signal_entropy(real_raw_signal)
        max_power_win_, freq_max_power_, mean_power_win_ = self.frequency_features(raw_signal_200us)

        pentropy = self.pentropy(real_raw_signal)
        pentropy_mean_ = np.mean(pentropy)
        pentropy_std_ = np.std(pentropy)
        
        features_list = [mean_, median_, std_, mad_, rms_, percentile_25th_, percentile_75th_,
                        iqr_, skewness_, kurtosis_, entropy_, max_power_win_, freq_max_power_,
                        mean_power_win_, pentropy_mean_, pentropy_std_]
        
        return features_list
    
    def extract_features_direct_optimized(self, raw_signal_200us):
        """
        Extract 16 features directly from in-memory signal.
        Optimized with vectorized math and redundant calculations removed.
        """
        real_raw_signal = raw_signal_200us.real if np.iscomplexobj(raw_signal_200us) else raw_signal_200us
        
        mean_ = np.mean(real_raw_signal)
        median_ = np.median(real_raw_signal)
        std_ = np.std(real_raw_signal)
        mad_ = np.mean(np.absolute(real_raw_signal - mean_))
        rms_ = np.sqrt(np.mean(real_raw_signal ** 2))
        
        percentiles = np.percentile(real_raw_signal, [25, 75])
        percentile_25th_ = percentiles[0]
        percentile_75th_ = percentiles[1]
        iqr_ = percentile_75th_ - percentile_25th_
        
        skewness_ = skew(real_raw_signal)
        kurtosis_ = kurtosis(real_raw_signal, fisher=False, bias=True)
        entropy_ = self.signal_entropy_optimized(real_raw_signal)
        
        freq_max_power_, max_power_win_, mean_power_win_ = self.frequency_features_optimized(real_raw_signal)

        pentropy = self.pentropy_optimized(real_raw_signal)
        pentropy_mean_ = np.mean(pentropy)
        pentropy_std_ = np.std(pentropy)
        
        features_list = [mean_, median_, std_, mad_, rms_, percentile_25th_, percentile_75th_,
                        iqr_, skewness_, kurtosis_, entropy_, max_power_win_, freq_max_power_,
                        mean_power_win_, pentropy_mean_, pentropy_std_]
        
        return features_list

    def frequency_features_optimized(self, raw_signal_):
        nfft = 512
        win = signal_sci.get_window('boxcar', nfft)
        raw_signal_ = raw_signal_ - np.mean(raw_signal_)
        freq, psd = signal_sci.welch(raw_signal_,
                                     fs=self.fs,
                                     nfft=nfft,
                                     window=win,
                                     noverlap=0, return_onesided=False)

        maxpower_win_ = np.max(psd)
        freq_maxpower_ = freq[np.argmax(psd)] 
        meanpower_win_ = np.mean(psd)
        
        return freq_maxpower_, maxpower_win_, meanpower_win_
    
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
    
    def histogram_signalEntropy_optimized(self, x, descriptor=None):
        if x.ndim != 1:
            raise ValueError("Invalid dimension of x")

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

        # Eliminato il ciclo for: usiamo vettorizzazione pura
        y = np.round((x - lower) / (upper - lower) * ncell + 0.5).astype(np.int32)
        valid_mask = (y >= 1) & (y <= ncell)
        
        # np.bincount è implementato in C ed è velocissimo
        counts = np.bincount(y[valid_mask], minlength=ncell + 1)
        result = counts[1:ncell+1].astype(float)

        return result, descriptor

    def signal_entropy_optimized(self, sig):
        h, descriptor = self.histogram_signalEntropy_optimized(sig, descriptor=None)
        lowerbound, upperbound, ncell = descriptor

        count = np.sum(h)
        if count == 0:
            return 0.0

        nz_mask = h > 0
        h_nz = h[nz_mask]
        logf = np.log(h_nz)

        estimate = -np.sum(h_nz * logf) / count
        sigma_sum = np.sum(h_nz * (logf ** 2))
        
        # FIX PER JETSON NANO: Evitare NaN causati da errori in virgola mobile negativi
        variance_estimate = max(0.0, (sigma_sum / count - estimate ** 2) / (count - 1))
        sigma = np.sqrt(variance_estimate) if count > 1 else 0

        estimate = estimate + np.log(count) + np.log((upperbound - lowerbound) / ncell)
        nbias = -(ncell - 1) / (2 * count)

        # unbiased estimate
        estimate = estimate - nbias
        base = np.e
        
        estimate = estimate / np.log(base)

        return estimate

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
    
    def pentropy_optimized(self, real_raw_signal_):
        nfft = 1024
        win = signal_sci.get_window(('kaiser', 20.0), 1024)

        _, _, Sxx = signal_sci.spectrogram((real_raw_signal_ / np.sum(win)) * math.sqrt(2), self.fs,
                                            nfft=2048,
                                            nperseg=1024,
                                            return_onesided=True,
                                            noverlap=993, scaling='spectrum')
                                            
        
        sum_Sxx = np.sum(Sxx, axis=0) + 1e-12 
        normalized_spectrum = Sxx / sum_Sxx  
        safe_p = np.where(normalized_spectrum > 0, normalized_spectrum, 1)
        entropy = np.sum(-normalized_spectrum * np.log2(safe_p), axis=0)
        
        entropy /= np.log2(normalized_spectrum.shape[0])
        return entropy
