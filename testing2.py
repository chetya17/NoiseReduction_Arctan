import torch
import torchaudio
from scipy.io.wavfile import write
import numpy as np
from datasets import load_dataset
import torch.nn as nn
from training import PositionalEncoding, LightweightTransformerBlock, NoiseSuppressionTransformer
# Load the trained model
def load_trained_model(model_path, device):
    model = NoiseSuppressionTransformer()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

#Convert spectrogram back to audio
def spectrogram_to_audio(spec, sample_rate=16000, n_fft=512, hop_length=256):
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    audio = griffin_lim(spec.sqrt())
    return audio.numpy()
def spectrogram_to_audio(spec, sample_rate=16000, n_fft=512, hop_length=256):
    
    if spec.shape[-1] != n_fft // 2 + 1:
        raise ValueError(f"Spectrogram has invalid frequency dimension {spec.shape[-1]}. Expected {n_fft // 2 + 1}.")
    
    # Ensure the input is real (if required)
    spec = spec.abs()
    
    # Griffin-Lim to convert spectrogram back to waveform
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    audio = griffin_lim(spec)
    return audio.numpy()

# Save audio to file
def save_audio(audio, file_name, sample_rate=16000):
    audio = audio / np.max(np.abs(audio))  # Normalize
    write(file_name, sample_rate, (audio * 32767).astype(np.int16))
    print(f"Saved: {file_name}")

# Main function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
    test_dataset = ds['test']

    # Load the trained model
    model_path = 'best_noise_suppression_model.pth'
    model = load_trained_model(model_path, device)

    # Take a random sample from the test dataset
    idx = np.random.randint(0, len(test_dataset))
    sample = test_dataset[idx]

    noisy_audio = torch.tensor(sample['noisy']['array'], dtype=torch.float32)
    clean_audio = torch.tensor(sample['clean']['array'], dtype=torch.float32)

    # Preprocess noisy audio to spectrogram
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256, power=2.0)
    noisy_spec = spec_transform(noisy_audio)

    # Model inference
    with torch.no_grad():
        noisy_spec = noisy_spec.transpose(0, 1).unsqueeze(0).to(device)  # [T, 257] -> [1, T, 257]
        cleaned_spec = model(noisy_spec)
        cleaned_spec = cleaned_spec.squeeze(0).transpose(0, 1).cpu()  # [1, T, 257] -> [T, 257]

    # Convert spectrograms back to audio
    print(f"noisy_spec shape before conversion: {noisy_spec.shape}")

    noisy_audio_output = spectrogram_to_audio(noisy_spec.squeeze(0).cpu())
    cleaned_audio_output = spectrogram_to_audio(cleaned_spec)
    clean_target_audio = spectrogram_to_audio(spec_transform(clean_audio))

    # Save audio files
    save_audio(noisy_audio_output, "noisy_audio.wav")
    save_audio(cleaned_audio_output, "cleaned_audio.wav")
    save_audio(clean_target_audio, "target_clean_audio.wav")

if __name__ == '__main__':
    main()
