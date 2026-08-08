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
We have prepared a wrapper script `srt2anki` that can be called from any folder. It accepts **one or more** files, and the model is loaded only once for the whole run:

```bash
# A single file
./srt2anki /path/to/your/file.srt

# Several specific files
./srt2anki lecture1.srt lecture3.srt

# A subset via a shell glob
./srt2anki lecture_*.srt
```

To process an **entire folder** at once, use `srt2anki-batch` (see below). Add `--context N` after the file list to override the context length for that run.

By default, a file is **skipped** if its `_notes.md` and `.apkg` already exist (handy for resuming an interrupted batch). Pass `--force` to regenerate and overwrite them:

```bash
./srt2anki lecture1.srt --force
```

**What will happen (for each file):**
1. The script automatically activates the required environment and loads the model.
2. The model generates detailed Markdown notes and saves them to a `_notes.md` file.
3. It automatically extracts the flashcards and saves them to a `.tsv` file.
4. **It generates a ready-to-use `.apkg` deck** that you can import into Anki with a double-click!

**Important Limitation (File Size):**
Language models have a strict context window limit. With the default context of 8192 (the base model's native window), this script can safely process up to ~6192 tokens of subtitles (the remaining 2000 tokens are reserved for the generated response), roughly 40-50 minutes of talking. 
The script features an **automatic safety check**: if your `.srt` file is too large, it will instantly cancel the process and tell you exactly how many tokens it counted. If this happens, simply split your file into smaller parts (e.g., `part1.srt`, `part2.srt`) and process them separately to prevent memory crashes.

### Batch Processing (Whole Folder)
To process every `.srt` file in a directory at once, use the `srt2anki-batch` wrapper. The model is loaded **only once** and applied to all files, which is far faster than calling `srt2anki` per file:

```bash
./srt2anki-batch /path/to/folder
```

If you omit the path, it processes `.srt` files in the current directory. Each file gets its own `_notes.md`, `.tsv`, and `.apkg`. A problem with one file (missing, too long, or a generation error) is isolated — the batch continues and prints a summary at the end. The `--context` flag works here too:

```bash
./srt2anki-batch /path/to/folder --context 4096
```

### 💡 Tip: Global Access
To avoid typing the full path to the scripts every time, you can create global `alias`es. Note that `srt2anki` and `srt2anki-batch` are two separate scripts, so each needs its own alias. Just run the block for your shell from the project folder:

**For Bash:**
```bash
echo "alias srt2anki=\"$(realpath srt2anki)\"" >> ~/.bashrc
echo "alias srt2anki-batch=\"$(realpath srt2anki-batch)\"" >> ~/.bashrc
source ~/.bashrc
```

**For Zsh (macOS default):**
```zsh
echo "alias srt2anki=\"$(realpath srt2anki)\"" >> ~/.zshrc
echo "alias srt2anki-batch=\"$(realpath srt2anki-batch)\"" >> ~/.zshrc
source ~/.zshrc
```

**For Fish:**
```fish
echo "alias srt2anki=\"$(realpath srt2anki)\"" >> ~/.config/fish/config.fish
echo "alias srt2anki-batch=\"$(realpath srt2anki-batch)\"" >> ~/.config/fish/config.fish
source ~/.config/fish/config.fish
```

Now you can process subtitles from any folder on your computer with a simple command:

```bash
srt2anki video.srt
```

### Alternative Run Methods
The `lora_model` folder contains standard HuggingFace PEFT weights. You can merge them with the base model and convert them into the GGUF format (for use with Ollama / LM Studio) using built-in Unsloth utilities if you ever need a fully standalone model file.

### Advanced Configuration (Context Size)
The recommended context is `8192` tokens — the native window of the base Llama-3-8B model. Going higher forces untrained RoPE scaling and degrades the quality of the generated notes and cards, so the limit is enforced differently depending on where the value comes from:

* **`config.json`:** values above `8192` are **rejected** — the script stops immediately (before loading the model) to catch accidental misconfiguration.
* **`--context` flag:** treated as a deliberate override — it is **allowed** to exceed `8192`, but prints a quality warning and continues. Use this if you knowingly want to trade quality for longer context.

You can **lower** the context to save VRAM (e.g. on smaller GPUs) via the `config.json` file in the project root:

```json
{
    "max_seq_length": 8192
}
```

**Approximate VRAM usage (4-bit quantization):**
* **`8192` (Default / Max):** ~8 GB VRAM. Safe for 8GB GPUs (RTX 3060/4060). Fits ~45-50 mins of video.
* **`4096`:** ~5-6 GB VRAM. For low-VRAM GPUs; process shorter clips or split your files.

To handle longer lectures, split the `.srt` into parts (`part1.srt`, `part2.srt`, …) and process them separately.

**Temporary Override via Command Line**
You can also override the configuration for a single run by passing the `--context` argument directly in the terminal. Unlike `config.json`, this flag may exceed `8192` (with a quality warning) if you deliberately want a longer context:
```bash
srt2anki video.srt --context 4096
```
If you pass this argument, it will completely ignore the value in `config.json` for that specific run.
