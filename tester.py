import torch
import torchaudio
from datasets import load_dataset
import random
import os
import numpy as np
from training import NoiseSuppressionTransformer, PositionalEncoding, LightweightTransformerBlock
def load_saved_model(model_path, device):
    # Initialize model with the same architecture
    model = NoiseSuppressionTransformer()
    
    # Load the saved state
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# def spectrogram_to_audio(spectrogram, length=None):
#     # Create Griffin-Lim transform for phase reconstruction
#     griffin_lim = torchaudio.transforms.GriffinLim(
#         n_fft=512,
#         hop_length=256,
#         power=2.0,
#         n_iter=32
#     )
    
#     # Ensure spectrogram is in the correct shape [freq, time]
#     if spectrogram.shape[0] != 257:
#         spectrogram = spectrogram.transpose(-1, -2)
    
#     # Convert spectrogram to audio using Griffin-Lim algorithm
#     waveform = griffin_lim(spectrogram)
    
#     # Trim to original length if provided
#     if length is not None:
#         waveform = waveform[:length]
    
#     return waveform
def spectrogram_to_audio(spectrogram, length=None):
    # Add small epsilon to avoid zero values
    print("Cleannnedddddddddddddddddddddddd spec",spectrogram)
    spectrogram = spectrogram + 1e-6
    
    # Create Griffin-Lim transform for phase reconstruction
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=512,
        hop_length=256,
        power=2.0,
        n_iter=32
    )
    
    # Ensure spectrogram is in the correct shape [freq, time]
    if spectrogram.shape[0] != 257:
        spectrogram = spectrogram.transpose(-1, -2)
    
    # Convert spectrogram to audio using Griffin-Lim algorithm
    waveform = griffin_lim(spectrogram)
    
    # Trim to original length if provided
    if length is not None:
        waveform = waveform[:length]
    
    return waveform
def process_random_sample():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the dataset
    ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
    
    # Load the saved model
    model = load_saved_model('best_noise_suppression_model.pth', device)
    
    # Create spectrogram transform
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=512,
        hop_length=256,
        power=2.0
    )
    
    # Randomly select a sample from the test set
    test_set = ds['test']
    random_idx = random.randint(0, len(test_set) - 1)
    sample = test_set[random_idx]
    
    # Get noisy and clean audio
    noisy_audio = torch.tensor(sample['noisy']['array'], dtype=torch.float32)
    clean_audio = torch.tensor(sample['clean']['array'], dtype=torch.float32)
    original_length = noisy_audio.shape[0]
    
    # Save original noisy and clean audio
    output_dir = 'audio_samples'
    os.makedirs(output_dir, exist_ok=True)
    
    torchaudio.save(
        os.path.join(output_dir, 'original_noisy.wav'),
        noisy_audio.unsqueeze(0),
        sample_rate=16000
    )
    
    torchaudio.save(
        os.path.join(output_dir, 'original_clean.wav'),
        clean_audio.unsqueeze(0),
        sample_rate=16000
    )
    
    # Process through model
    with torch.no_grad():
        # Convert to spectrogram - shape will be [freq, time]
        noisy_spec = spec_transform(noisy_audio)
        clean_spec = spec_transform(clean_audio)
        print("Noisy speccccccccccccccccccccccccccccccc",noisy_spec)
        print("Clean speccccccccccccccccccccccccccccccc",clean_spec)
        # Add batch dimension [1, freq, time]
        noisy_spec = noisy_spec.unsqueeze(0).to(device)
        
        # Process through model
        denoised_spec = model(noisy_spec)
        
        # Move back to CPU and remove batch dimension
        denoised_spec = denoised_spec.cpu().squeeze(0)
        
        # Convert back to audio using Griffin-Lim
        denoised_audio = spectrogram_to_audio(denoised_spec, original_length)
        
        # Normalize audio
        denoised_audio = denoised_audio / (torch.max(torch.abs(denoised_audio)) + 1e-8)
        
        # Save denoised audio
        torchaudio.save(
            os.path.join(output_dir, 'denoised.wav'),
            denoised_audio.unsqueeze(0),
            sample_rate=16000
        )
    
    print(f"Files saved in {output_dir}:")
    print("1. original_noisy.wav - The original noisy audio")
    print("2. original_clean.wav - The target clean audio")
    print("3. denoised.wav - The model's denoised output")
    
    return os.path.join(output_dir)

if __name__ == '__main__':
    output_path = process_random_sample()
    print(f"\nAll files have been saved to: {output_path}")