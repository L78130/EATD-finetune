import numpy as np
import librosa
import torch
import laion_clap
import os

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

model = laion_clap.CLAP_Module(enable_fusion=False).to(device)
model.load_ckpt() # download the default pretrained checkpoint.

# Get audio embeddings from audio data
#audio_data, _ = librosa.load('F:/datasets/EATD-Corpus/EATD-Corpus/t_1/negative.wav', sr=48000) # sample rate should be 48000
#audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
#audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float() # quantize before send it in to the model
#audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
#print(audio_embed[:,-20:])
#print(audio_embed.shape)

# Get text embedings from texts, but return torch tensor:
#text_data = ["I love the contrastive learning", "I love the pretrain model"] 
#text_embed = model.get_text_embedding(text_data, use_tensor=True)
#print(text_embed)
#print(text_embed.shape)

#combined = torch.cat([audio_embed, text_embed], dim=1)  # shape: (1, 1024)
#print(combined.shape)

# Iterate over all subdirectories in the current path
for folder_name in os.listdir('F:\datasets\EATD-Corpus\EATD-Corpus'):
    folder_path = os.path.join('F:\datasets\EATD-Corpus\EATD-Corpus', folder_name)

    # Process only if it's a directory
    if not os.path.isdir(folder_path):
        continue

    # Construct paths to the files
    prompt_path = os.path.join(folder_path, 'neutral.txt')
    audio_path = os.path.join(folder_path, 'neutral_out.wav')

    # Skip if the required files don't exist
    if not os.path.exists(prompt_path) or not os.path.exists(audio_path):
        continue

    print(f"Processing folder: {folder_name}")

    # Load prompt from a text file, specifying UTF-8 encoding
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    # Load audio from a WAV file
    audio_data, _ = librosa.load(audio_path, sr=48000)
    
    if audio_data.size == 0:
        print(f"  - Using dummy audio for {folder_name}")
        audio_data = np.zeros(48000, dtype=np.float32)  # 1 second of silence at 48 kHz

    audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
    audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float() # quantize before send it in to the model
    audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
    print(audio_embed.shape)

    print(audio_embed.shape)

    # Get text embedings from texts, but return torch tensor:
    text_embed = model.get_text_embedding(prompt, use_tensor=True)
    print(text_embed.shape)

    combined = torch.cat([audio_embed, text_embed], dim=1)
    print(combined.shape)
    # Define the output folder path
    output_folder_path = os.path.join('F:\\datasets\\EATD-Corpus\\CLAP', folder_name)

    # Ensure the output directory exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Save the last hidden state tensor in the respective folder
    output_path = os.path.join(output_folder_path, 'nue_hidden_state.pt')
    torch.save(combined, output_path)
    print(f"  - Saved last hidden state to {output_path}")