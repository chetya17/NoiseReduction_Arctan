import numpy as np
import torch
import torchaudio
from scipy import signal
from scipy.io.wavfile import write


class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_profile = None

        # Load the pre-trained VAD model
        self.vad_model = self.load_vad_model()

    def load_vad_model(self):
        """
        Load pre-trained VAD model from torchaudio hub.
        """
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )
        return model

    def estimate_noise_profile(self, audio_data):
        """
        Estimate the noise profile from the provided audio data.
        """
        print("Estimating noise profile...")
        _, noise_psd = signal.welch(audio_data, self.sample_rate, nperseg=512)
        self.noise_profile = noise_psd

    def spectral_subtraction(self, audio_chunk):
        """
        Perform spectral subtraction to reduce noise.
        """
        if self.noise_profile is None:
            return audio_chunk

        _, current_psd = signal.welch(audio_chunk, self.sample_rate, nperseg=512)
        cleaned_psd = np.maximum(current_psd - self.noise_profile, 0)
        _, cleaned_audio = signal.istft(np.sqrt(cleaned_psd))
        return cleaned_audio

    def detect_voice_activity(self, audio_chunk):
        """
        Detect voice activity using the VAD model.
        """
        if len(audio_chunk) != 512:
            raise ValueError(f"Audio chunk must have 512 samples, but got {len(audio_chunk)} samples.")

        tensor_chunk = torch.FloatTensor(audio_chunk).unsqueeze(0)
        return self.vad_model(tensor_chunk, self.sample_rate).item() > 0.5

    def process_audio_file(self, input_path, output_path):
        """
        Process a noisy audio file and save the cleaned audio.
        """
        print(f"Loading audio from {input_path}...")
        waveform, sample_rate = torchaudio.load(input_path)
        if sample_rate != self.sample_rate:
            print(f"Resampling audio to {self.sample_rate} Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        audio_data = waveform.squeeze().numpy()

        # Estimate noise profile
        self.estimate_noise_profile(audio_data)

        # Process audio in chunks
        print("Processing audio...")
        chunk_size = 512
        cleaned_audio = []

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]

            # Pad the chunk if it's shorter than 512 samples
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")

            has_voice = self.detect_voice_activity(chunk)
            if has_voice:
                cleaned_chunk = self.spectral_subtraction(chunk)
            else:
                cleaned_chunk = np.zeros_like(chunk)
            cleaned_audio.extend(cleaned_chunk)

        # Save the cleaned audio
        cleaned_audio = np.array(cleaned_audio)
        print(f"Saving cleaned audio to {output_path}...")
        write(output_path, self.sample_rate, (cleaned_audio * 32767).astype(np.int16))


# Example Usage
if __name__ == "__main__":
    processor = AudioProcessor()
    input_audio_path = "original_noisy.wav"
    output_audio_path = "cleaned_audio2.wav"
    processor.process_audio_file(input_audio_path, output_audio_path)
