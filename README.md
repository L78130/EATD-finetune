# 🎧 Audio-Text Model Training
---
### Models used
Qwen2-audio-7B(used bitsandbytes config to avoid GPU memory exceeding   
CLAP LAION-630k-audioset-best.pt  
AudioFlamingo2-0.5B

### Setup
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
#note requirements vary for flamingo and CLAP from Qwen
```
```bash
run python /your_dir/qwen_train.py
run python /your_dir/CLAP_train.py
run python /your_dir/AFLA_train.py
```
### Model performance
| Model | Performance |
|:------|:-------------|
| Qwen2 | 75.11% |
| CLAP  | 81.86% |
| Model | 77.61% |



