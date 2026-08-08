# SRT to MD & Anki (Finetuned Llama-3-8B)

This repository contains a finetuned model that automatically converts SRT subtitles into detailed Markdown notes and Anki flashcards (TSV and APKG formats).

> **⚠️ Important Note:** This specific LoRA adapter was finetuned exclusively on **Russian** subtitles and is designed to output Russian notes and flashcards. If you feed it English subtitles, it will likely act as a translator and output the notes in Russian.

## Folder Contents:
- `lora_model/` — Trained weights (LoRA adapters) for the Llama-3-8B-Instruct model (downloaded separately).
- `inference.py` — Ready-to-use script for model inference.
- `finetune.py` — Original training script used to finetune this model (for reference).
- `train_dataset.jsonl` — The dataset used for training.

## How to use the model?

Thanks to the automated wrapper, running the model is incredibly simple. You can generate notes, TSV tables, and ready-to-use `.apkg` decks for Anki with a single command.

### 1. Download Model Weights (lora_model)
Since model weights take up a lot of space, they are not stored in this Git repository.
Download the `lora_model.zip` archive from [this Google Drive link](https://drive.google.com/file/d/1wUPzqhBhXbxa65ejb-bMxPrxSIMkwMZe/view?usp=sharing) and extract it so that a `lora_model/` folder with files appears in the root of the project.

### 2. Environment Setup (Install Dependencies)
Open a terminal in the project folder and run these commands once:

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies at once
pip install -r requirements.txt
```

### 3. Run Generation
We have prepared a wrapper script `srt2anki` that can be called from any folder:

```bash
./srt2anki /path/to/your/file.srt
```

**What will happen:**
1. The script automatically activates the required environment and loads the model.
2. The model generates detailed Markdown notes and saves them to a `_notes.md` file.
3. It automatically extracts the flashcards and saves them to a `.tsv` file.
4. **It generates a ready-to-use `.apkg` deck** that you can import into Anki with a double-click!

### 💡 Tip: Global Access
To avoid typing the full path to the script every time, you can create a global `alias`. Just run this command from the project folder to automatically add the correct absolute path to your `~/.bashrc`:

```bash
echo "alias srt2anki=\"$(realpath srt2anki)\"" >> ~/.bashrc
source ~/.bashrc
```

Now you can process subtitles from any folder on your computer with a simple command:

```bash
srt2anki video.srt
```

### Alternative Run Methods
The `lora_model` folder contains standard HuggingFace PEFT weights. You can merge them with the base model and convert them into the GGUF format (for use with Ollama / LM Studio) using built-in Unsloth utilities if you ever need a fully standalone model file.
