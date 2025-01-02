import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from datasets import load_dataset
import numpy as np
import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.device_count())  # Should return the number of CUDA devices
print(torch.cuda.current_device()) 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LightweightTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        return src + self.dropout(src2)

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
#         # x shape: [batch, freq, time] -> [batch, time, freq]
#         x = x.transpose(-1, -2)
        
#         # Project and process
#         x = self.input_projection(x)
#         x = self.pos_encoder(x)
        
#         for block in self.transformer_blocks:
#             x = block(x)
            
#         x = self.output_projection(x)
        
#         # Return to original shape: [batch, time, freq] -> [batch, freq, time]
#         x = x.transpose(-1, -2)
#         return x
class NoiseSuppressionTransformer(nn.Module):
    def __init__(self, input_size=257, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            LightweightTransformerBlock(d_model, nhead)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, input_size)
        
        # Add magnitude scaling layers
        self.input_scale = nn.Parameter(torch.ones(1))
        self.output_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # Store original magnitude
        mag = torch.mean(torch.abs(x))
        
        # Scale input
        x = x * self.input_scale
        
        # x shape: [batch, freq, time] -> [batch, time, freq]
        x = x.transpose(-1, -2)
        
        # Project and process
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.output_projection(x)
        
        # Return to original shape: [batch, time, freq] -> [batch, freq, time]
        x = x.transpose(-1, -2)
        
        # Scale output back to original magnitude range
        x = x * self.output_scale * mag
        
        return x
class VoiceBankDataset(Dataset):
    def __init__(self, dataset, split='train', segment_length=16000):
        super().__init__()
        self.data = dataset[split]
        self.segment_length = segment_length
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=512,
            hop_length=256,
            power=2.0
        )
        # Store the length of the dataset
        self.length = len(self.data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of bounds")
            
        # Get audio data
        noisy_audio = torch.tensor(self.data[idx]['noisy']['array'], dtype=torch.float32)
        clean_audio = torch.tensor(self.data[idx]['clean']['array'], dtype=torch.float32)
        
        # Handle length
        if noisy_audio.shape[0] > self.segment_length:
            start = torch.randint(0, noisy_audio.shape[0] - self.segment_length, (1,))
            noisy_audio = noisy_audio[start:start + self.segment_length]
            clean_audio = clean_audio[start:start + self.segment_length]
        else:
            noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_length - noisy_audio.shape[0]))
            clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_length - clean_audio.shape[0]))
        
        # Convert to spectrograms - shape will be [freq, time]
        noisy_spec = self.spec_transform(noisy_audio)
        clean_spec = self.spec_transform(clean_audio)
        
        return noisy_spec, clean_spec

# def train_model(model, train_loader, valid_loader, num_epochs=100, device='cuda'):
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
#     model = model.to(device)
#     best_loss = float('inf')
    
#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         train_loss = 0
        
#         for batch_idx, (noisy, clean) in enumerate(train_loader):
#             noisy, clean = noisy.to(device), clean.to(device)
            
#             optimizer.zero_grad()
#             output = model(noisy)
#             loss = criterion(output, clean)
            
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
            
#             if batch_idx % 50 == 0:
#                 print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
#         avg_train_loss = train_loss / len(train_loader)
        
#         # Validation phase
#         model.eval()
#         valid_loss = 0
        
#         with torch.no_grad():
#             for noisy, clean in valid_loader:
#                 noisy, clean = noisy.to(device), clean.to(device)
#                 output = model(noisy)
#                 valid_loss += criterion(output, clean).item()
        
#         avg_valid_loss = valid_loss / len(valid_loader)
#         scheduler.step(avg_valid_loss)
        
#         print(f'Epoch {epoch}:')
#         print(f'  Training Loss: {avg_train_loss:.6f}')
#         print(f'  Validation Loss: {avg_valid_loss:.6f}')
        
#         # Save best model
#         if avg_valid_loss < best_loss:
#             best_loss = avg_valid_loss
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': best_loss,
#             }, 'best_noise_suppression_model.pth')
def train_model(model, train_loader, valid_loader, num_epochs=100, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    model = model.to(device)
    best_loss = float('inf')
    
    # Add logging of spectrograms
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            # Log spectrogram statistics
            if batch_idx == 0:
                print(f"\nBatch statistics:")
                print(f"Noisy range: {noisy.min():.3f} to {noisy.max():.3f}")
                print(f"Clean range: {clean.min():.3f} to {clean.max():.3f}")
            
            optimizer.zero_grad()
            output = model(noisy)
            
            # Log output statistics
            if batch_idx == 0:
                print(f"Output range: {output.min():.3f} to {output.max():.3f}")
            
            loss = criterion(output, clean)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation with detailed logging
        model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for noisy, clean in valid_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                valid_loss += criterion(output, clean).item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        scheduler.step(avg_valid_loss)
        
        print(f'Epoch {epoch}:')
        print(f'  Training Loss: {avg_train_loss:.6f}')
        print(f'  Validation Loss: {avg_valid_loss:.6f}')
        
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_noise_suppression_model.pth')
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading VoiceBank-DEMAND dataset...")
    ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
    
    # Create datasets
    train_dataset = VoiceBankDataset(ds, split='train')
    valid_dataset = VoiceBankDataset(ds, split='test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, num_workers=4)
    
    # Initialize model
    model = NoiseSuppressionTransformer()
    
    # Train the model
    print("Starting training...")
    train_model(model, train_loader, valid_loader, num_epochs=10, device=device)

if __name__ == '__main__':
    main()