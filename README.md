# SRT to MD & Anki (Finetuned Llama-3.1-8B)

This repository contains a finetuned model that automatically converts SRT subtitles into detailed Markdown notes and Anki flashcards (TSV and APKG formats). The current adapter is finetuned on **Llama-3.1-8B-Instruct** (native 128k context) on a combined `srt`+`txt` dataset.

> **⚠️ Important Note:** This specific LoRA adapter was finetuned exclusively on **Russian** subtitles and is designed to output Russian notes and flashcards. If you feed it English subtitles, it will likely act as a translator and output the notes in Russian.

## Folder Contents
- `lora_model_llama31/` — Trained weights (LoRA adapter) for **Llama-3.1-8B-Instruct** (downloaded separately). `inference.py` uses this automatically if present.
- `lora_model/` — Older adapter for the original Llama-3-8B (fallback, used only if `lora_model_llama31/` is absent).
- `inference.py` — The inference script (all the logic lives here).
- `srt2anki` — Wrapper to process one or more files.
- `srt2anki-batch` — Wrapper to process every `.srt` in a folder.
- `config.json` — Default context length.
- `finetune.py` — Training script (Llama-3.1 base, combined dataset).
- `build_combined_dataset.py` — Builds the combined `srt`+`txt` training set from `train_dataset.jsonl`.
- `train_dataset.jsonl` / `train_dataset_combined.jsonl` — Source dataset / the combined (srt+txt) set used for training.
- `srt_converter_flat` / `.c` — Standalone C tool that flattens `.srt` to plain `.txt` (its cleanup is also built into `--format txt`).

---

## Setup (once)

### 1. Download Model Weights
Model weights are not stored in Git. Download `lora_model_llama31.zip` from [this Google Drive link](https://drive.google.com/file/d/18F4d65b66jKoDkfl5pmQMK8J9_scJals/view?usp=sharing) and extract it so a `lora_model_llama31/` folder appears in the project root.

> **Note:** The old Llama-3 archive lives at [this Google Drive link](https://drive.google.com/file/d/1wUPzqhBhXbxa65ejb-bMxPrxSIMkwMZe/view?usp=sharing) and extracts to `lora_model/` (used only as a fallback if `lora_model_llama31/` is absent).

### 2. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) Global aliases
So you can call the scripts from any folder. `srt2anki` and `srt2anki-batch` are two separate scripts, so each needs its own alias — run the block for your shell from the project folder:

```bash
# Bash
echo "alias srt2anki=\"$(realpath srt2anki)\"" >> ~/.bashrc
echo "alias srt2anki-batch=\"$(realpath srt2anki-batch)\"" >> ~/.bashrc
source ~/.bashrc

# Zsh
echo "alias srt2anki=\"$(realpath srt2anki)\"" >> ~/.zshrc
echo "alias srt2anki-batch=\"$(realpath srt2anki-batch)\"" >> ~/.zshrc
source ~/.zshrc

# Fish
echo "alias srt2anki=\"$(realpath srt2anki)\"" >> ~/.config/fish/config.fish
echo "alias srt2anki-batch=\"$(realpath srt2anki-batch)\"" >> ~/.config/fish/config.fish
source ~/.config/fish/config.fish
```

---

## Commands

There are two entry points. Both load the model **only once** per run and accept the same optional flags.

| Command | Input | Use it for |
|---------|-------|-----------|
| `srt2anki <files...> [flags]` | one or more `.srt` files | a single file, a hand-picked list, or a shell glob |
| `srt2anki-batch [dir] [flags]` | a directory (default: current) | every `.srt` in a folder at once |

```bash
srt2anki lecture.srt                 # one file
srt2anki lecture1.srt lecture3.srt   # several specific files
srt2anki lecture_*.srt               # a subset via a shell glob
srt2anki-batch                       # all .srt in the current directory
srt2anki-batch /path/to/folder       # all .srt in a given directory
```

## Parameters

| Parameter | Values | Default | What it does |
|-----------|--------|---------|--------------|
| positional | file paths (`srt2anki`) or a directory (`srt2anki-batch`) | — | What to process. `srt2anki-batch` with no path uses the current directory. |
| `--format` | `srt` \| `txt` | `srt` | **`srt`**: feeds raw subtitles; timecode-aware prompt (model can flag unclear moments as `⚠️ [... @ <TIMECODE>]`). **`txt`**: flattens the file first (removes indices, timecodes, HTML tags), giving far fewer splits, and uses a prompt that never mentions timecodes. |
| `--context N` | integer | value from `config.json` (16384) | Overrides context length for this run only. No hard ceiling — a very large value only prints a VRAM warning; see [Context & limits](#context--limits). |
| `--max-new-tokens N` | integer | value from `config.json` (2000) | Caps how many tokens the model generates per part (response length). **Higher** = more detailed notes/more cards, but slower and eats more context budget; **lower** = faster. Must be smaller than the context. |
| `--force` | flag | off | Regenerate even if outputs exist. Without it, a file whose `_notes.md` **and** `.apkg` already exist is skipped (great for resuming an interrupted batch). |
| `--recursive`, `-r` | flag | off | **`srt2anki-batch` only.** Also scan subfolders for `.srt` files, not just the top level. |

Flags go **after** the files/directory and can be combined:

```bash
srt2anki-batch /path/to/folder --format txt --force
srt2anki lecture.srt --context 4096
```

### Running in the background & saving a log

Progress is printed live (timestamped, flushed immediately), so for a quick run you can just run the command and watch the terminal. To run it detached and save a log, the syntax differs by shell — and **quote any path that contains spaces**:

```bash
# Bash / Zsh
srt2anki-batch "/path/with spaces/folder" --format txt > run.log 2>&1 &
tail -f run.log
```

```fish
# Fish — use &> for "stdout+stderr to file"
srt2anki-batch "/path/with spaces/folder" --format txt &>run.log &
tail -f run.log

# Or see it live AND save it at the same time:
srt2anki-batch "/path/with spaces/folder" --format txt &| tee run.log
```

> By default `srt2anki-batch` scans only the **top level** of a folder. To include nested subfolders, add `--recursive` (or `-r`):
> ```bash
> srt2anki-batch "/path/to/course" --recursive --format txt
> ```

## Results (what you get)

For each input `file.srt`, up to three files are written **next to the original** (same folder):

| Output | When | Contents |
|--------|------|----------|
| `file_notes.md` | always | Detailed Markdown summary (H2/H3 structure). In `srt` mode may include `⚠️` timecode notes. Ends with a ` ```tsv ``` ` block of the cards. |
| `file.tsv` | if cards were produced | Tab-separated `question<TAB>answer` pairs (no header) — import into anything. |
| `file.apkg` | if cards were produced & `genanki` installed | Ready-to-import Anki deck — double-click to add it to Anki. |

Notes on behavior:
- **Long files are handled automatically.** If a file exceeds the token budget, it is split on subtitle boundaries into as many parts as needed; the parts are generated separately and then **merged** into a single `_notes.md` and a single `.apkg` per original file (part sections are marked `## Part N`).
- **Reliable card generation.** To avoid missing flashcards, each part uses a three-layer strategy: (1) a `repetition_penalty` to prevent degeneration loops, (2) a deterministic greedy pass, then up to 2 sampled retries if no `tsv` block appears, and (3) a **cross-format fallback** — if a file still produces no cards, the whole file is retried in the opposite format (`srt`↔`txt`).
- **Live progress log.** Every step is printed with a timestamp and flushed immediately (loading, per-part generation, which attempt/format is running now, saves), so you can `tail -f` a redirected log and see exactly what's happening.
- **Batch runs are resilient.** A file that is missing, unparseable, or errors out is skipped; the rest continue, and a summary (`done: X/Y …`) is printed at the end. Exit code is non-zero if anything failed.

---

## Context & limits

The base model (**Llama-3.1-8B**) is native to **128k tokens**, so there is **no quality cliff** from raising the context — unlike the old Llama-3 (which was capped at 8192). The real constraint is now **VRAM**: the KV cache grows ~128 KB per token.

- **Default:** `16384` (set in `config.json`). Comfortable on a 16 GB GPU.
- **No hard ceiling.** A large `--context`/`config.json` value is allowed; above ~32768 the script only prints a **VRAM warning** (CUDA out-of-memory becomes likely on 16 GB). Nothing is blocked.

Edit `config.json` to change the defaults:

```json
{
    "max_seq_length": 16384,
    "max_new_tokens": 2000
}
```

`max_new_tokens` is the cap on generated tokens per part (the answer length), reserved from the context budget — so the subtitles for one part must fit in `max_seq_length − max_new_tokens`. Raise it for longer, more detailed notes (slower); lower it for speed. Override per-run with `--max-new-tokens`.

**Approximate limits on a 16 GB GPU (4-bit weights ~6 GB):**
- **16384 (default):** ~2 GB KV cache — comfortable.
- **~32768:** ~4 GB KV cache — the practical maximum here.
- **128k:** would need ~16 GB of KV cache alone — not feasible on 16 GB (the model *supports* it with more VRAM).

Files that don't fit the current context are split and merged automatically (see above), so you rarely need to raise it.

---

## Retraining
The current adapter was trained with `finetune.py` on `train_dataset_combined.jsonl` (built by `build_combined_dataset.py`, which produces one `srt` and one `txt` example per source lecture — the `txt` targets have their timecode notes stripped). It saves to `lora_model_llama31/`. Note: with a small dataset the model mostly learns the **output format**; broader quality needs more source lectures.

## Alternative Run Methods
The adapter folder contains standard HuggingFace PEFT weights. You can merge them with the base model and convert to GGUF (for Ollama / LM Studio) using built-in Unsloth utilities if you ever need a fully standalone model file.
