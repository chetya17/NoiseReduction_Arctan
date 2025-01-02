#RAW AUDIO PLOTTING,NO NOISE REDUCTION.

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

# Setup matplotlib for real-time spectrogram plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Initialize spectrogram placeholders
x = np.arange(CHUNK)
y = np.zeros(CHUNK)
line1, = ax1.plot(x, y, label="Input Audio")
line2, = ax2.plot(x, y, label="Output Audio")

ax1.set_title("Input Audio Spectrogram")
ax2.set_title("Output Audio Spectrogram")
ax1.set_ylim(-32768, 32768)
ax2.set_ylim(-32768, 32768)

ax1.legend()
ax2.legend()

# Latency tracking
latencies = []

def update(frame):
    global latencies

    # Record start time
    start_time = time.time()

    # Read input audio
    input_audio = input_stream.read(CHUNK, exception_on_overflow=False)

    # Record after processing start
    output_audio = input_audio  # (Direct pass-through in this example)

    # Record end time
    end_time = time.time()

    # Measure latency
    latency = (end_time - start_time) * 1000  # Convert to ms
    latencies.append(latency)

    # Update spectrograms
    input_data = np.frombuffer(input_audio, dtype=np.int16)
    output_data = np.frombuffer(output_audio, dtype=np.int16)

    line1.set_ydata(input_data)
    line2.set_ydata(output_data)

    # Output the processed audio
    output_stream.write(output_audio)

    return line1, line2

# Animate the spectrogram
ani = FuncAnimation(fig, update, interval=1, blit=True)

print("Streaming audio with real-time spectrograms. Press Ctrl+C to stop.")
try:
    plt.show()
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
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"Average latency: {avg_latency:.2f} ms")
    print("Audio streaming stopped.")
