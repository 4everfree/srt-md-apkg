import sys
import os
import re
import json
import time
import random
import argparse
from unsloth import FastLanguageModel


def log(msg):
    """Timestamped, immediately-flushed progress line (so logs show what's
    happening *now*, even when stdout is redirected to a file)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# The base model (Llama-3.1-8B) has a native context window of 128k tokens, so
# there is no quality cliff — the practical limit is VRAM (the KV cache grows
# ~128 KB/token). We only warn past this soft threshold; nothing is blocked.
DEFAULT_MAX_SEQ_LENGTH = 16384
MAX_NEW_TOKENS = 2000
VRAM_WARN_SEQ_LENGTH = 32768  # above this, OOM is likely on a 16 GB GPU


def build_messages(content, srt_file, fmt):
    # DO NOT TRANSLATE THE PROMPTS! The LoRA adapter was trained on this exact Russian structure.
    if fmt == "txt":
        # Plain-text input: no timecodes exist, so the prompt must not ask for them.
        prompt = f"""Ты — эксперт в своей области и внимательный слушатель. Твоя задача: прочитать текст лекции ниже и создать подробный конспект в формате Markdown.
В конспекте должны быть отражены все нюансы, примеры и объяснения из лекции. Структурируй текст с помощью заголовков H2 (##) и H3 (###).

После конспекта составь список вопросов и ответов для закрепления материала (Anki).
ОБЯЗАТЕЛЬНО помести эти вопросы и ответы в блок кода `tsv`, разделяя вопрос и ответ знаком табуляции, без заголовков столбцов.

Текст лекции:
{content}"""
    else:
        # Raw SRT input: timecodes are present, so the model may reference them.
        prompt = f"""Ты — эксперт в своей области и внимательный слушатель. Твоя задача: прочитать субтитры ниже и создать подробный конспект в формате Markdown.
В конспекте должны быть отражены все нюансы, примеры и объяснения из лекции. Структурируй текст с помощью заголовков H2 (##) и H3 (###).
Если какие-то моменты непонятны без видеоряда, укажи это в формате: > ⚠️ **[Неясно в {os.path.basename(srt_file)} @ <ТАЙМКОД>]:** <комментарий>.

После конспекта составь список вопросов и ответов для закрепления материала (Anki).
ОБЯЗАТЕЛЬНО помести эти вопросы и ответы в блок кода `tsv`, разделяя вопрос и ответ знаком табуляции, без заголовков столбцов.

Субтитры:
{content}"""
    return [{"role": "user", "content": prompt}]


def count_prompt_tokens(tokenizer, content, srt_file, fmt):
    """Token count of the full chat-templated prompt for the given content."""
    ids = tokenizer.apply_chat_template(
        build_messages(content, srt_file, fmt),
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(ids)


def clean_block_to_text(block):
    """Flatten one SRT block to plain text: drop index/timecode lines and HTML
    tags (mirrors the srt_converter_flat.c preprocessing)."""
    out_lines = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.isdigit():          # subtitle index line
            continue
        if "-->" in line:           # timecode line
            continue
        line = re.sub(r'<[^>]*>', '', line).strip()  # strip HTML/formatting tags
        if line:
            out_lines.append(line)
    return " ".join(out_lines)


def parse_blocks(srt_text, fmt):
    """Return (blocks, joiner) for chunking. In 'srt' mode blocks are the raw
    subtitle blocks; in 'txt' mode they are cleaned plain-text units."""
    raw = [b for b in re.split(r'\n\s*\n', srt_text.strip()) if b.strip()]
    if fmt == "txt":
        units = [t for t in (clean_block_to_text(b) for b in raw) if t]
        return units, " "
    return raw, "\n\n"


def chunk_blocks(blocks, joiner, tokenizer, srt_file, fmt, max_input_tokens):
    """Greedily pack blocks (on block boundaries) so each chunk's prompt fits
    within max_input_tokens."""
    chunks = []
    current = []
    for block in blocks:
        candidate = joiner.join(current + [block])
        if count_prompt_tokens(tokenizer, candidate, srt_file, fmt) > max_input_tokens and current:
            chunks.append(joiner.join(current))
            current = [block]
        else:
            current.append(block)
    if current:
        chunks.append(joiner.join(current))

    # A single block that alone exceeds the budget can't be split further here.
    for i, chunk in enumerate(chunks):
        if count_prompt_tokens(tokenizer, chunk, srt_file, fmt) > max_input_tokens:
            print(f"[!] WARNING: chunk {i + 1} still exceeds {max_input_tokens} tokens "
                  f"(a single block is too long); generation may be truncated.")
    return chunks


def _generate_once(content, srt_file, model, tokenizer, fmt, sample):
    """One generation pass. repetition_penalty guards against degeneration loops
    (e.g. endless '* * *'); greedy is deterministic, sampling is the retry fallback."""
    inputs = tokenizer.apply_chat_template(
        build_messages(content, srt_file, fmt),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, use_cache=True, repetition_penalty=1.15)
    if sample:
        gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    outputs = model.generate(input_ids=inputs, **gen_kwargs)
    result = tokenizer.batch_decode(outputs)[0]
    result = result.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip()
    return result


def run_model_on_text(content, srt_file, model, tokenizer, fmt, retries=2):
    """Generate notes for one chunk, reliably. First a deterministic greedy pass;
    if it yields no ```tsv``` card block, retry a few times with sampling to recover it."""
    log("      generating (greedy pass)...")
    result = _generate_once(content, srt_file, model, tokenizer, fmt, sample=False)
    if split_notes_and_tsv(result)[1]:
        log("      cards found on greedy pass.")
        return result
    for attempt in range(1, retries + 1):
        log(f"      no cards yet — sampled retry {attempt}/{retries}...")
        alt = _generate_once(content, srt_file, model, tokenizer, fmt, sample=True)
        if split_notes_and_tsv(alt)[1]:
            log(f"      recovered cards on sampled retry {attempt}.")
            return alt
    log("      [!] no tsv block after retries — keeping best-effort output.")
    return result


def split_notes_and_tsv(result):
    """Separate the Markdown notes from the ```tsv``` card block in a response."""
    tsv_match = re.search(r'```tsv\n(.*?)```', result, re.DOTALL | re.IGNORECASE)
    tsv_content = tsv_match.group(1).strip() if tsv_match else ""
    # Remove the tsv code block from the notes body so it isn't duplicated.
    notes_body = re.sub(r'```tsv\n.*?```', '', result, flags=re.DOTALL | re.IGNORECASE).strip()
    return notes_body, tsv_content


def _process_in_format(srt_file, srt_text, model, tokenizer, max_seq_length, fmt):
    """Process the whole file in one format. Returns (merged_md, combined_tsv);
    merged_md is None if the subtitles are empty/unparseable."""
    max_input_tokens = max_seq_length - MAX_NEW_TOKENS
    blocks, joiner = parse_blocks(srt_text, fmt)
    if not blocks:
        return None, ""

    chunks = chunk_blocks(blocks, joiner, tokenizer, srt_file, fmt, max_input_tokens)
    if len(chunks) > 1:
        total_tokens = count_prompt_tokens(tokenizer, joiner.join(blocks), srt_file, fmt)
        log(f"  file is long ({total_tokens} tokens, format={fmt}) — splitting into {len(chunks)} part(s) "
            f"to fit the {max_input_tokens}-token budget.")

    notes_parts = []
    tsv_lines = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_tokens = count_prompt_tokens(tokenizer, chunk, srt_file, fmt)
        log(f"  [{fmt}] part {i}/{len(chunks)} ({chunk_tokens} tokens)...")
        result = run_model_on_text(chunk, srt_file, model, tokenizer, fmt)
        notes_body, tsv_content = split_notes_and_tsv(result)
        if notes_body:
            header = f"## Part {i}\n\n" if len(chunks) > 1 else ""
            notes_parts.append(header + notes_body)
        if tsv_content:
            tsv_lines.extend(line for line in tsv_content.split('\n') if '\t' in line)

    combined_tsv = "\n".join(tsv_lines)
    merged_md = "\n\n---\n\n".join(notes_parts)
    if combined_tsv:
        merged_md += "\n\n```tsv\n" + combined_tsv + "\n```\n"
    return merged_md, combined_tsv


def generate_notes(srt_file, model, tokenizer, max_seq_length, force=False, fmt="srt"):
    # Strip only the extension, so a ".srt" substring elsewhere in the path
    # (e.g. a folder named ".srtbackup") is never touched.
    base_path, _ = os.path.splitext(srt_file)
    output_md = base_path + "_notes.md"
    output_tsv = base_path + ".tsv"
    output_apkg = base_path + ".apkg"

    # Skip files that were already fully processed, unless --force is given.
    # Checked before generation so we don't waste time re-running the model.
    if not force and os.path.exists(output_md) and os.path.exists(output_apkg):
        log(f"[=] output already exists, skipping (use --force to overwrite): {os.path.basename(srt_file)}")
        return True

    with open(srt_file, "r", encoding="utf-8") as f:
        srt_text = f.read()

    log(f"processing in '{fmt}' mode...")
    merged_md, combined_tsv = _process_in_format(srt_file, srt_text, model, tokenizer, max_seq_length, fmt)
    if merged_md is None:
        log(f"[-] empty or unparseable subtitles: {srt_file}")
        return False

    # Cross-format fallback: if this format produced no cards, retry the whole
    # file in the opposite format (srt<->txt) — often recovers a missing tsv block.
    if not combined_tsv:
        opposite = "txt" if fmt == "srt" else "srt"
        log(f"[!] no cards in '{fmt}' mode — retrying whole file in '{opposite}' mode...")
        md2, tsv2 = _process_in_format(srt_file, srt_text, model, tokenizer, max_seq_length, opposite)
        if tsv2:
            merged_md, combined_tsv = md2, tsv2
            log(f"[+] recovered cards via '{opposite}' mode.")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(merged_md)
    log(f"[+] notes saved: {output_md}")

    if not combined_tsv:
        log("[-] no tsv cards even after cross-format fallback; .tsv/.apkg not generated.")
        return True

    with open(output_tsv, "w", encoding="utf-8") as f:
        f.write(combined_tsv)
    log(f"[+] tsv saved: {output_tsv}")

    try:
        import genanki
        deck_name = os.path.splitext(os.path.basename(srt_file))[0]
        deck_id = random.randrange(1 << 30, 1 << 31)
        model_id = random.randrange(1 << 30, 1 << 31)

        anki_model = genanki.Model(
          model_id,
          'QA Model',
          fields=[{'name': 'Question'}, {'name': 'Answer'}],
          templates=[{
            'name': 'Card 1',
            'qfmt': '{{Question}}',
            'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
          }])

        anki_deck = genanki.Deck(deck_id, deck_name)
        for line in combined_tsv.split('\n'):
            if '\t' in line:
                q, a = line.split('\t', 1)
                anki_deck.add_note(genanki.Note(model=anki_model, fields=[q.strip(), a.strip()]))

        genanki.Package(anki_deck).write_to_file(output_apkg)
        log(f"[+] anki deck saved: {output_apkg} ({len(anki_deck.notes)} cards)")
    except ImportError:
        log("[-] 'genanki' not installed — .apkg not generated.")

    return True


def main():
    parser = argparse.ArgumentParser(description="SRT to Markdown & Anki")
    parser.add_argument("srt_files", nargs="+", help="Path(s) to one or more .srt files")
    parser.add_argument("--context", type=int, default=None, help="Override context length (max_seq_length)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files instead of skipping")
    parser.add_argument("--format", choices=["srt", "txt"], default="srt",
                        help="Input handling: 'srt' feeds raw subtitles (timecode-aware prompt); "
                             "'txt' flattens to plain text (indices/timecodes/tags removed, no-timecode prompt)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Prefer the newer Llama-3.1 adapter once it exists; fall back to the old one.
    # Each adapter pulls its own base model, so no other change is needed.
    model_path = os.path.join(script_dir, "lora_model_llama31")
    if not os.path.isdir(model_path):
        model_path = os.path.join(script_dir, "lora_model")
    print(f"Using adapter: {os.path.basename(model_path)}")

    # Load configuration
    config_path = os.path.join(script_dir, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            max_seq_length = config.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)
    except FileNotFoundError:
        max_seq_length = DEFAULT_MAX_SEQ_LENGTH

    # Command-line override wins over config.json.
    if args.context is not None:
        max_seq_length = args.context

    # No hard ceiling: Llama-3.1 is native to 128k, so quality doesn't degrade.
    # The real constraint is VRAM (KV cache), so we only warn about likely OOM.
    if max_seq_length > VRAM_WARN_SEQ_LENGTH:
        print(f"\n[!] WARNING: context {max_seq_length} is large — the KV cache grows ~128 KB/token,")
        print(f"    so anything above ~{VRAM_WARN_SEQ_LENGTH} risks CUDA out-of-memory on a 16 GB GPU. Watch VRAM.")

    log(f"using max_seq_length: {max_seq_length} | input format: {args.format}")

    # Model settings
    dtype = None
    load_in_4bit = True

    log("loading model (this can take a minute)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path, # Load our trained weights using absolute path
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Enable fast inference
    FastLanguageModel.for_inference(model)
    log("model loaded.")

    # Process every file with the model loaded once. A failure on one file
    # (missing, too long, or a runtime error) is isolated so the batch keeps going.
    total = len(args.srt_files)
    succeeded = 0
    failed = []
    for i, srt_file in enumerate(args.srt_files, start=1):
        log("#" * 60)
        log(f"# [{i}/{total}] {os.path.basename(srt_file)}")

        if not os.path.exists(srt_file):
            log(f"[-] file not found, skipping: {srt_file}")
            failed.append(srt_file)
            continue

        try:
            if generate_notes(srt_file, model, tokenizer, max_seq_length, force=args.force, fmt=args.format):
                succeeded += 1
            else:
                failed.append(srt_file)
        except Exception as e:
            log(f"[-] error processing {srt_file}: {e}")
            failed.append(srt_file)

    log("=" * 60)
    log(f"done: {succeeded}/{total} file(s) processed successfully.")
    if failed:
        log(f"failed/skipped ({len(failed)}):")
        for f in failed:
            log(f"  - {f}")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
