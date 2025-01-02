import numpy as np
import sounddevice as sd
import threading
import time
from scipy import signal
import scipy.signal as signal
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from collections import deque

class NoiseReducer:
    def __init__(self, sample_rate=44100, block_size=2048, n_fft=2048, 
                 noise_reduce_strength=0.8, signal_boost=False, boost_factor=1.5):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.n_fft = n_fft
        self.noise_reduce_strength = noise_reduce_strength
        self.signal_boost = signal_boost
        self.boost_factor = boost_factor
        
        # Initialize noise profile
        self.noise_profile = None
        self.calibrated = False
        
        # Latency measurement
        self.processing_times = deque(maxlen=100)
        
        # Visualization queues
        self.input_queue = Queue()
        self.output_queue = Queue()
        
        # Overlap-add buffers
        self.prev_input = np.zeros(self.block_size // 2)
        self.prev_output = np.zeros(self.block_size // 2)
        
        # Window function
        self.window = signal.hann(self.block_size)
        
    def calibrate_noise(self, duration=2):
        """Record ambient noise for calibration"""
        print("Recording noise profile... Please ensure silence.")
        noise_data = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, channels=1)
        sd.wait()
        
        # Calculate average noise spectrum
        _, _, noise_specs = signal.stft(noise_data.flatten(), 
                                      fs=self.sample_rate,
                                      nperseg=self.n_fft)
        self.noise_profile = np.mean(np.abs(noise_specs), axis=1)
        self.calibrated = True
        print("Noise profile calibrated.")
    
    def process_block(self, input_block):
        """Process a single block of audio"""
        start_time = time.time()
        
        # Apply window function
        windowed = input_block * self.window
        
        # STFT
        _, _, Zxx = signal.stft(windowed, fs=self.sample_rate, 
                               nperseg=self.n_fft, noverlap=self.n_fft//2)
        
        # Spectral subtraction
        if self.calibrated:
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            # Subtract noise profile
            mag_cleaned = np.maximum(
                mag - self.noise_reduce_strength * self.noise_profile[:, np.newaxis],
                0
            )
            
            # Optional signal boost
            if self.signal_boost:
                mag_cleaned = mag_cleaned * self.boost_factor
            
            # Reconstruct signal
            Zxx_cleaned = mag_cleaned * np.exp(1j * phase)
        else:
            Zxx_cleaned = Zxx
        
        # Inverse STFT
        _, output = signal.istft(Zxx_cleaned, fs=self.sample_rate,
                               nperseg=self.n_fft, noverlap=self.n_fft//2)
        
        # Measure processing time
        end_time = time.time()
        self.processing_times.append(end_time - start_time)
        
        # Update visualization queues
        if not self.input_queue.full():
            self.input_queue.put(input_block)
        if not self.output_queue.full():
            self.output_queue.put(output)
        
        return output

def visualize_spectrograms(input_queue, output_queue):
    """Separate process for visualization"""
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    while True:
        try:
            input_data = input_queue.get_nowait()
            output_data = output_queue.get_nowait()
            
            # Update spectrograms
            ax1.clear()
            ax2.clear()
            
            # Input spectrogram
            f, t, Sxx = signal.spectrogram(input_data, fs=44100)
            ax1.pcolormesh(t, f, 10 * np.log10(Sxx))
            ax1.set_ylabel('Frequency [Hz]')
            ax1.set_title('Input Spectrogram')
            
            # Output spectrogram
            f, t, Sxx = signal.spectrogram(output_data, fs=44100)
            ax2.pcolormesh(t, f, 10 * np.log10(Sxx))
            ax2.set_ylabel('Frequency [Hz]')
            ax2.set_xlabel('Time [sec]')
            ax2.set_title('Output Spectrogram')
            
            plt.pause(0.01)
            
        except:
            plt.pause(0.1)

def main():
    # Initialize noise reducer
    noise_reducer = NoiseReducer()
    
    # Calibrate noise profile
    noise_reducer.calibrate_noise()
    
    # Start visualization in separate process
    viz_process = Process(target=visualize_spectrograms, 
                         args=(noise_reducer.input_queue, noise_reducer.output_queue))
    viz_process.start()
    
    def audio_callback(indata, outdata, frames, time, status):
        """Real-time audio callback"""
        if status:
            print(status)
        
        # Process audio
        output = noise_reducer.process_block(indata[:, 0])
        outdata[:] = output.reshape(-1, 1)
    
    # Start audio stream
    with sd.Stream(channels=1, callback=audio_callback, 
                  samplerate=noise_reducer.sample_rate,
                  blocksize=noise_reducer.block_size):
        print("Noise reduction active. Press Ctrl+C to stop.")
        
        try:
            while True:
                # Print average latency every second
                time.sleep(1)
                avg_latency = np.mean(noise_reducer.processing_times) * 1000
                print(f"Average processing latency: {avg_latency:.2f} ms")
                
        except KeyboardInterrupt:
            print("\nStopping...")
    
    # Clean up
    viz_process.terminate()

if __name__ == "__main__":
    main()