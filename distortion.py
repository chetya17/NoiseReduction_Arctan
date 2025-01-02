##NOTES - Horrible voice quality,not sure if noise is being reduced or just the volume.
import pyaudio
import numpy as np
import noisereduce as nr
import time

# Parameters
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate
CHUNK = 1024  # Number of samples per chunk

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open audio streams for input and output
input_stream = audio.open(format=FORMAT, 
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)

output_stream = audio.open(format=FORMAT, 
                           channels=CHANNELS,
                           rate=RATE,
                           output=True,
                           frames_per_buffer=CHUNK)

# Latency tracking
latencies = []

def noise_reduction(input_audio, rate):
    """Apply noise reduction to audio data."""
    audio_data = np.frombuffer(input_audio, dtype=np.int16).astype(np.float32)
    reduced_noise = nr.reduce_noise(y=audio_data, sr=rate, n_std_thresh_stationary=1.5, stationary=True)
    return reduced_noise.astype(np.int16).tobytes()

print("Streaming audio with noise reduction and latency measurement. Press Ctrl+C to stop.")
try:
    while True:
        # Record start time
        start_time = time.time()

        # Read input audio
        input_audio = input_stream.read(CHUNK, exception_on_overflow=False)

        # Apply noise reduction
        output_audio = noise_reduction(input_audio, RATE)

        # Record end time
        end_time = time.time()

        # Measure latency
        latency = (end_time - start_time) * 1000  # Convert to ms
        latencies.append(latency)

        # Output the processed audio
        output_stream.write(output_audio)

except KeyboardInterrupt:
    print("\nStopping audio streaming...")

finally:
    # Stop and close streams
    input_stream.stop_stream()
    input_stream.close()
    output_stream.stop_stream()
    output_stream.close()
    audio.terminate()

    # Print latency statistics
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"Average latency: {avg_latency:.2f} ms")
    else:
        print("No latency data recorded.")
    print("Audio streaming stopped.")






# import pyaudio
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import noisereduce as nr
# import time

# # Parameters
# FORMAT = pyaudio.paInt16  # 16-bit resolution
# CHANNELS = 1  # Mono audio
# RATE = 44100  # Sampling rate
# CHUNK = 1024  # Number of samples per chunk

# # Initialize PyAudio
# audio = pyaudio.PyAudio()

# # Open audio streams for input and output
# input_stream = audio.open(format=FORMAT, 
#                           channels=CHANNELS,
#                           rate=RATE,
#                           input=True,
#                           frames_per_buffer=CHUNK)

# output_stream = audio.open(format=FORMAT, 
#                            channels=CHANNELS,
#                            rate=RATE,
#                            output=True,
#                            frames_per_buffer=CHUNK)

# # Setup matplotlib for real-time spectrogram plotting
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# # Initialize spectrogram placeholders
# x = np.arange(CHUNK)
# y = np.zeros(CHUNK)
# line1, = ax1.plot(x, y, label="Input Audio")
# line2, = ax2.plot(x, y, label="Output Audio")

# ax1.set_title("Input Audio Spectrogram")
# ax2.set_title("Output Audio Spectrogram")
# ax1.set_ylim(-32768, 32768)
# ax2.set_ylim(-32768, 32768)

# ax1.legend()
# ax2.legend()

# # Latency tracking
# latencies = []

# def noise_reduction(input_audio, rate):
#     """Apply noise reduction to audio data."""
#     audio_data = np.frombuffer(input_audio, dtype=np.int16).astype(np.float32)
#     reduced_noise = nr.reduce_noise(y=audio_data, sr=rate, n_std_thresh_stationary=1.5, stationary=True)
#     return reduced_noise.astype(np.int16).tobytes()

# def update(frame):
#     global latencies

#     # Record start time
#     start_time = time.time()

#     # Read input audio
#     input_audio = input_stream.read(CHUNK, exception_on_overflow=False)

#     # Apply noise reduction
#     output_audio = noise_reduction(input_audio, RATE)

#     # Record end time
#     end_time = time.time()

#     # Measure latency
#     latency = (end_time - start_time) * 1000  # Convert to ms
#     latencies.append(latency)

#     # Update spectrograms
#     input_data = np.frombuffer(input_audio, dtype=np.int16)
#     output_data = np.frombuffer(output_audio, dtype=np.int16)

#     line1.set_ydata(input_data)
#     line2.set_ydata(output_data)

#     # Output the processed audio
#     output_stream.write(output_audio)

#     return line1, line2

# # # Animate the spectrogram
# # ani = FuncAnimation(fig, update, interval=1, blit=True)

# print("Streaming audio with noise reduction, real-time spectrograms, and latency measurement. Press Ctrl+C to stop.")
# try:
#     plt.show()
# except KeyboardInterrupt:
#     print("\nStopping audio streaming...")

# finally:
#     # Stop and close streams
#     input_stream.stop_stream()
#     input_stream.close()
#     output_stream.stop_stream()
#     output_stream.close()
#     audio.terminate()

#     # Print latency statistics
#     avg_latency = sum(latencies) / len(latencies) if latencies else 0
#     print(f"Average latency: {avg_latency:.2f} ms")
#     print("Audio streaming stopped.")












# import pyaudio
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import time

# # Parameters
# FORMAT = pyaudio.paInt16  # 16-bit resolution
# CHANNELS = 1  # Mono audio
# RATE = 44100  # Sampling rate
# CHUNK = 1024  # Number of samples per chunk

# # Initialize PyAudio
# audio = pyaudio.PyAudio()

# # Open audio streams for input and output
# input_stream = audio.open(format=FORMAT, 
#                           channels=CHANNELS,
#                           rate=RATE,
#                           input=True,
#                           frames_per_buffer=CHUNK)

# output_stream = audio.open(format=FORMAT, 
#                            channels=CHANNELS,
#                            rate=RATE,
#                            output=True,
#                            frames_per_buffer=CHUNK)

# # Setup matplotlib for real-time spectrogram plotting
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# # Initialize spectrogram placeholders
# x = np.arange(CHUNK)
# y = np.zeros(CHUNK)
# line1, = ax1.plot(x, y, label="Input Audio")
# line2, = ax2.plot(x, y, label="Output Audio")

# ax1.set_title("Input Audio Spectrogram")
# ax2.set_title("Output Audio Spectrogram")
# ax1.set_ylim(-32768, 32768)
# ax2.set_ylim(-32768, 32768)

# ax1.legend()
# ax2.legend()

# # Latency tracking
# latencies = []

# def update(frame):
#     global latencies

#     # Record start time
#     start_time = time.time()

#     # Read input audio
#     input_audio = input_stream.read(CHUNK, exception_on_overflow=False)

#     # Record after processing start
#     output_audio = input_audio  # (Direct pass-through in this example)

#     # Record end time
#     end_time = time.time()

#     # Measure latency
#     latency = (end_time - start_time) * 1000  # Convert to ms
#     latencies.append(latency)

#     # Update spectrograms
#     input_data = np.frombuffer(input_audio, dtype=np.int16)
#     output_data = np.frombuffer(output_audio, dtype=np.int16)

#     line1.set_ydata(input_data)
#     line2.set_ydata(output_data)

#     # Output the processed audio
#     output_stream.write(output_audio)

#     return line1, line2

# # Animate the spectrogram
# ani = FuncAnimation(fig, update, interval=1, blit=True)

# print("Streaming audio with real-time spectrograms. Press Ctrl+C to stop.")
# try:
#     plt.show()
# except KeyboardInterrupt:
#     print("\nStopping audio streaming...")

# finally:
#     # Stop and close streams
#     input_stream.stop_stream()
#     input_stream.close()
#     output_stream.stop_stream()
#     output_stream.close()
#     audio.terminate()

#     # Print latency statistics
#     avg_latency = sum(latencies) / len(latencies) if latencies else 0
#     print(f"Average latency: {avg_latency:.2f} ms")
#     print("Audio streaming stopped.")

# NOISE REDUCTION
#  import pyaudio
# import numpy as np
# import noisereduce as nr

# # Parameters
# FORMAT = pyaudio.paInt16  # 16-bit resolution
# CHANNELS = 1  # Mono audio
# RATE = 16000  # Sampling rate
# CHUNK = 1024  # Number of samples per chunk

# # Initialize PyAudio
# audio = pyaudio.PyAudio()

# # Define noise reduction function
# def noise_reduction(input_data, rate):
#     # Convert byte data to numpy array
#     audio_data = np.frombuffer(input_data, dtype=np.int16).astype(np.float32)
    
#     # Apply noise reduction (stationary noise)
#     reduced_noise = nr.reduce_noise(y=audio_data, sr=rate, n_std_thresh_stationary=1.5, stationary=True)
    
#     # Convert back to int16 for playback
#     return reduced_noise.astype(np.int16).tobytes()

# # Open audio streams for input and output
# input_stream = audio.open(format=FORMAT, 
#                           channels=CHANNELS,
#                           rate=RATE,
#                           input=True,
#                           frames_per_buffer=CHUNK)

# output_stream = audio.open(format=FORMAT, 
#                            channels=CHANNELS,
#                            rate=RATE,
#                            output=True,
#                            frames_per_buffer=CHUNK)

# print("Real-time noise reduction running. Speak into the microphone...")

# try:
#     while True:
#         # Read a chunk of audio data from the microphone
#         input_audio = input_stream.read(CHUNK, exception_on_overflow=False)
        
#         # Reduce noise in the audio
#         output_audio = noise_reduction(input_audio, RATE)
        
#         # Play the noise-reduced audio through the speakers
#         output_stream.write(output_audio)

# except KeyboardInterrupt:
#     print("\nStopping...")

# finally:
#     # Stop and close the streams
#     input_stream.stop_stream()
#     input_stream.close()
#     output_stream.stop_stream()
#     output_stream.close()
#     audio.terminate()

# print("Real-time noise reduction stopped.")

#-------------------------------------------------------------------------------------
#RAW AUDIO
# import pyaudio

# # Parameters
# FORMAT = pyaudio.paInt16  # 16-bit resolution
# CHANNELS = 1  # Mono audio
# RATE = 44100  # Sampling rate (standard for audio)
# CHUNK = 1024  # Number of samples per chunk

# # Initialize PyAudio
# audio = pyaudio.PyAudio()

# # Open audio streams for input and output
# input_stream = audio.open(format=FORMAT, 
#                           channels=CHANNELS,
#                           rate=RATE,
#                           input=True,
#                           frames_per_buffer=CHUNK)

# output_stream = audio.open(format=FORMAT, 
#                            channels=CHANNELS,
#                            rate=RATE,
#                            output=True,
#                            frames_per_buffer=CHUNK)

# print("Streaming raw audio. Speak into the microphone... Press Ctrl+C to stop.")

# try:
#     while True:
#         # Read raw audio data from microphone
#         input_audio = input_stream.read(CHUNK, exception_on_overflow=False)
        
#         # Output raw audio directly to speakers
#         output_stream.write(input_audio)

# except KeyboardInterrupt:
#     print("\nStopping audio streaming...")

# finally:
#     # Stop and close streams
#     input_stream.stop_stream()
#     input_stream.close()
#     output_stream.stop_stream()
#     output_stream.close()
#     audio.terminate()

# print("Audio streaming stopped.")

#---------------------------------------------------------------------
# import torch
# import torchaudio
# import torchaudio.transforms as T
# import sounddevice as sd
# from scipy.io.wavfile import write
# import numpy as np
# import torch.nn as nn
# from training import PositionalEncoding, LightweightTransformerBlock

# class NoiseSuppressionTransformer(nn.Module):
#     def __init__(self, input_size=257, d_model=64, nhead=4, num_layers=3):
#         super().__init__()
#         self.input_projection = nn.Linear(input_size, d_model)
#         self.pos_encoder = PositionalEncoding(d_model)
#         self.transformer_blocks = nn.ModuleList([
#             LightweightTransformerBlock(d_model, nhead)
#             for _ in range(num_layers)
#         ])
#         self.output_projection = nn.Linear(d_model, input_size)
        
#     def forward(self, x):
#         x = self.input_projection(x)
#         x = self.pos_encoder(x)
#         for block in self.transformer_blocks:
#             x = block(x)
#         return self.output_projection(x)

# def microphone_to_spectrogram(audio, sample_rate=16000):
#     # Convert waveform to spectrogram
#     spec_transform = T.Spectrogram(n_fft=512, hop_length=256, power=2.0)
#     return spec_transform(torch.tensor(audio, dtype=torch.float32))

# def spectrogram_to_audio(spectrogram, sample_rate=16000):
#     # Convert spectrogram back to waveform
#     griffin_lim = T.GriffinLim(n_fft=512, hop_length=256)
#     return griffin_lim(spectrogram)

# def main():
#     # Parameters
#     model_path = 'best_noise_suppression_model.pth'
#     sample_rate = 16000
#     duration = 5  # Record for 5 seconds
#     output_file = 'cleaned_audio.wav'

#     # Load the trained model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = NoiseSuppressionTransformer()
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval().to(device)
    
#     # Record audio from microphone
#     print("Recording...")
#     audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
#     sd.wait()
#     audio = audio.flatten()
#     print("Recording complete.")

#     # Convert audio to spectrogram
#     noisy_spec = microphone_to_spectrogram(audio)

#     # Normalize and process through model
#     noisy_spec = noisy_spec.transpose(0, 1).unsqueeze(0).to(device)  # Shape: [1, T, F]
#     with torch.no_grad():
#         clean_spec = model(noisy_spec)
#     clean_spec = clean_spec.squeeze(0).transpose(0, 1).cpu()  # Back to [F, T]

#     # Convert spectrogram back to audio
#     clean_audio = spectrogram_to_audio(clean_spec)
#     clean_audio = clean_audio.numpy()

#     # Save the cleaned audio
#     write(output_file, sample_rate, (clean_audio * 32767).astype(np.int16))
#     print(f"Cleaned audio saved to {output_file}")

# if __name__ == '__main__':
#     main()
# import torch
# import torchaudio
# from scipy.io.wavfile import write
# import numpy as np
# from datasets import load_dataset
# import torch.nn as nn
# from training import PositionalEncoding, LightweightTransformerBlock, NoiseSuppressionTransformer
# # Load the trained model
# def load_trained_model(model_path, device):
#     model = NoiseSuppressionTransformer()
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     model.eval()
#     return model

# Convert spectrogram back to audio
# def spectrogram_to_audio(spec, sample_rate=16000, n_fft=512, hop_length=256):
#     griffin_lim = torchaudio.transforms.GriffinLim(
#         n_fft=n_fft, hop_length=hop_length, power=2.0
#     )
#     audio = griffin_lim(spec.sqrt())
#     return audio.numpy()
# def spectrogram_to_audio(spec, sample_rate=16000, n_fft=512, hop_length=256):
    
#     if spec.shape[-1] != n_fft // 2 + 1:
#         raise ValueError(f"Spectrogram has invalid frequency dimension {spec.shape[-1]}. Expected {n_fft // 2 + 1}.")
    
#     # Ensure the input is real (if required)
#     spec = spec.abs()
    
#     # Griffin-Lim to convert spectrogram back to waveform
#     griffin_lim = torchaudio.transforms.GriffinLim(
#         n_fft=n_fft, hop_length=hop_length, power=2.0
#     )
#     audio = griffin_lim(spec)
#     return audio.numpy()

# # Save audio to file
# def save_audio(audio, file_name, sample_rate=16000):
#     audio = audio / np.max(np.abs(audio))  # Normalize
#     write(file_name, sample_rate, (audio * 32767).astype(np.int16))
#     print(f"Saved: {file_name}")

# # Main function
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # Load dataset
#     ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
#     test_dataset = ds['test']

#     # Load the trained model
#     model_path = 'best_noise_suppression_model.pth'
#     model = load_trained_model(model_path, device)

#     # Take a random sample from the test dataset
#     idx = np.random.randint(0, len(test_dataset))
#     sample = test_dataset[idx]

#     noisy_audio = torch.tensor(sample['noisy']['array'], dtype=torch.float32)
#     clean_audio = torch.tensor(sample['clean']['array'], dtype=torch.float32)

#     # Preprocess noisy audio to spectrogram
#     spec_transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256, power=2.0)
#     noisy_spec = spec_transform(noisy_audio)

#     # Model inference
#     with torch.no_grad():
#         noisy_spec = noisy_spec.transpose(0, 1).unsqueeze(0).to(device)  # [T, 257] -> [1, T, 257]
#         cleaned_spec = model(noisy_spec)
#         cleaned_spec = cleaned_spec.squeeze(0).transpose(0, 1).cpu()  # [1, T, 257] -> [T, 257]

#     # Convert spectrograms back to audio
#     print(f"noisy_spec shape before conversion: {noisy_spec.shape}")

#     noisy_audio_output = spectrogram_to_audio(noisy_spec.squeeze(0).cpu())
#     cleaned_audio_output = spectrogram_to_audio(cleaned_spec)
#     clean_target_audio = spectrogram_to_audio(spec_transform(clean_audio))

#     # Save audio files
#     save_audio(noisy_audio_output, "noisy_audio.wav")
#     save_audio(cleaned_audio_output, "cleaned_audio.wav")
#     save_audio(clean_target_audio, "target_clean_audio.wav")

# if __name__ == '__main__':
#     main()
