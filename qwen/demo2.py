from io import BytesIO
from urllib.request import urlopen
import librosa
import torch
import modelscope
import transformers

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#model_dir = modelscope.snapshot_download('Qwen/Qwen2-Audio-7B', local_dir="F:\models\QWEN")

bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type='nf4')

model = transformers.Qwen2AudioForConditionalGeneration.from_pretrained("F:\models\QWEN", torch_dtype=torch.bfloat16, quantization_config=bnb_config).to(device)
processor = transformers.AutoProcessor.from_pretrained("F:\models\QWEN")

prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"

audio, sr = librosa.load(BytesIO(urlopen(url).read()), sr=processor.feature_extractor.sampling_rate)
inputs = processor(text=prompt, audios=audio, return_tensors="pt")
inputs = inputs.to(model.device).to(model.dtype)

# Get hidden states by calling the model directly
with torch.no_grad():
    model_output = model(**inputs, output_hidden_states=True)
    hidden = model_output.hidden_states[-1]

# Generate token ids for the response
generated_ids = model.generate(**inputs, max_length=256)
generated_ids = generated_ids[:, inputs.input_ids.size(1):]
response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print("Last hidden state shape:", hidden.shape)
print("Response:", response)