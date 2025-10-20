import os
from urllib.request import urlopen
import librosa
import modelscope
import transformers
import torch

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#model_dir = modelscope.snapshot_download('Qwen/Qwen2-Audio-7B', local_dir="F:\models\QWEN")

bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type='nf4')

model = transformers.Qwen2AudioForConditionalGeneration.from_pretrained("F:\models\QWEN", torch_dtype=torch.bfloat16, quantization_config=bnb_config).to(device)
processor = transformers.AutoProcessor.from_pretrained("F:\models\QWEN")

# Iterate over all subdirectories in the current path
for folder_name in os.listdir('F:\datasets\EATD-Corpus\EATD-Corpus'):
    folder_path = os.path.join('F:\datasets\EATD-Corpus\EATD-Corpus', folder_name)

    # Process only if it's a directory
    if not os.path.isdir(folder_path):
        continue

    # Construct paths to the files
    prompt_path = os.path.join(folder_path, 'negative.txt')
    audio_path = os.path.join(folder_path, 'negative_out.wav')

    # Skip if the required files don't exist
    if not os.path.exists(prompt_path) or not os.path.exists(audio_path):
        continue

    print(f"Processing folder: {folder_name}")

    # Load prompt from a text file, specifying UTF-8 encoding
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    # Load audio from a WAV file
    audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

    # Process only the audio to get input features
    input = processor(text = prompt, audios = audio, return_tensors="pt", padding=True)
    input = input.to(model.device).to(model.dtype)
    
    # Get the last hidden state from the audio encoder
    with torch.no_grad():
        output = model(**input, output_hidden_states=True)
        last_hidden_state = output.hidden_states[-1]

    print(f"  - Last hidden state shape: {last_hidden_state.shape}")

    # Define the output folder path
    output_folder_path = os.path.join('F:\\datasets\\EATD-Corpus\\qwen', folder_name)

    # Ensure the output directory exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Save the last hidden state tensor in the respective folder
    output_path = os.path.join(output_folder_path, 'neg_hidden_state.pt')
    torch.save(last_hidden_state, output_path)
    print(f"  - Saved last hidden state to {output_path}")
