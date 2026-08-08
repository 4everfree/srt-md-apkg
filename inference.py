import sys
import os
import json
import argparse
from unsloth import FastLanguageModel

# The base model (Llama-3-8B) has a native context window of 8192 tokens.
# Going above this triggers untrained RoPE scaling and severely degrades quality,
# so 8192 is treated as a hard ceiling (see the check in main()).
DEFAULT_MAX_SEQ_LENGTH = 8192
MAX_NEW_TOKENS = 2000

def generate_notes(srt_file, model, tokenizer, max_seq_length, force=False):
    # Strip only the extension, so a ".srt" substring elsewhere in the path
    # (e.g. a folder named ".srtbackup") is never touched.
    base_path, _ = os.path.splitext(srt_file)
    output_md = base_path + "_notes.md"
    output_apkg = base_path + ".apkg"

    # Skip files that were already fully processed, unless --force is given.
    # Checked before generation so we don't waste time re-running the model.
    if not force and os.path.exists(output_md) and os.path.exists(output_apkg):
        print(f"[=] Output already exists, skipping (use --force to overwrite): {srt_file}")
        return True

    with open(srt_file, "r", encoding="utf-8") as f:
        srt_text = f.read()

    # DO NOT TRANSLATE THE PROMPT! The LoRA adapter was trained on this exact Russian prompt structure.
    prompt = f"""Ты — эксперт в своей области и внимательный слушатель. Твоя задача: прочитать субтитры ниже и создать подробный конспект в формате Markdown.
В конспекте должны быть отражены все нюансы, примеры и объяснения из лекции. Структурируй текст с помощью заголовков H2 (##) и H3 (###).
Если какие-то моменты непонятны без видеоряда, укажи это в формате: > ⚠️ **[Неясно в {os.path.basename(srt_file)} @ <ТАЙМКОД>]:** <комментарий>.

После конспекта составь список вопросов и ответов для закрепления материала (Anki). 
ОБЯЗАТЕЛЬНО помести эти вопросы и ответы в блок кода `tsv`, разделяя вопрос и ответ знаком табуляции, без заголовков столбцов.

Субтитры:
{srt_text}"""
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    token_count = inputs.shape[1]
    # Reserve tokens for the generated response
    max_input_tokens = max_seq_length - MAX_NEW_TOKENS
    
    if token_count > max_input_tokens:
        print(f"\n[-] ERROR: The file is too long! ({token_count} tokens).")
        print(f"Maximum allowed for subtitles is {max_input_tokens} tokens.")
        print("Please split your .srt file into smaller parts and try again.")
        return False

    inputs = inputs.to("cuda")
    
    print(f"Input size: {token_count} tokens. Generating response...")
    outputs = model.generate(input_ids=inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
    
    # Decode and clean up output
    result = tokenizer.batch_decode(outputs)[0]
    result = result.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip()
    
    print("\n" + "="*50 + "\n")
    print(result)
    
    import re
    import random

    # Save the result to a Markdown file
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n[+] Notes saved to: {output_md}")
    
    # Extract the TSV block and save to files
    tsv_match = re.search(r'```tsv\n(.*?)```', result, re.DOTALL | re.IGNORECASE)
    if tsv_match:
        tsv_content = tsv_match.group(1).strip()
        
        output_tsv = base_path + ".tsv"
        with open(output_tsv, "w", encoding="utf-8") as f:
            f.write(tsv_content)
        print(f"[+] TSV table saved to: {output_tsv}")
        
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
            
            for line in tsv_content.split('\n'):
                if '\t' in line:
                    q, a = line.split('\t', 1)
                    anki_deck.add_note(genanki.Note(model=anki_model, fields=[q.strip(), a.strip()]))
                    
            genanki.Package(anki_deck).write_to_file(output_apkg)
            print(f"[+] Anki deck saved to: {output_apkg}")
        except ImportError:
            print("[-] 'genanki' library is not installed. APKG file was not generated.")
    else:
        print("[-] TSV block not found in the generated response.")

    return True

def main():
    parser = argparse.ArgumentParser(description="SRT to Markdown & Anki")
    parser.add_argument("srt_files", nargs="+", help="Path(s) to one or more .srt files")
    parser.add_argument("--context", type=int, default=None, help="Override context length (max_seq_length)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files instead of skipping")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "lora_model")

    # Load configuration
    config_path = os.path.join(script_dir, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            max_seq_length = config.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)
    except FileNotFoundError:
        max_seq_length = DEFAULT_MAX_SEQ_LENGTH

    if args.context is not None:
        # Explicit --context is a deliberate "I know what I'm doing" override:
        # it is allowed to exceed the native ceiling, but we warn about quality.
        max_seq_length = args.context
        if max_seq_length > DEFAULT_MAX_SEQ_LENGTH:
            print(f"\n[!] WARNING: --context {max_seq_length} exceeds the base model's native {DEFAULT_MAX_SEQ_LENGTH} tokens.")
            print("    Untrained RoPE scaling kicks in above this — expect degraded notes/cards quality.")
    else:
        # Value came from config.json (or the default). Enforce the hard ceiling
        # here to catch accidental misconfiguration — before loading the model
        # or consuming any VRAM.
        if max_seq_length > DEFAULT_MAX_SEQ_LENGTH:
            print(f"\n[-] ERROR: config.json context ({max_seq_length}) exceeds the maximum of {DEFAULT_MAX_SEQ_LENGTH} tokens.")
            print("The base model is native to 8192 tokens; going higher wrecks output quality.")
            print("Lower max_seq_length in config.json, or pass --context explicitly to override on purpose.")
            sys.exit(1)

    print(f"Using max_seq_length: {max_seq_length}")

    # Model settings
    dtype = None
    load_in_4bit = True

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path, # Load our trained weights using absolute path
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Enable fast inference
    FastLanguageModel.for_inference(model)

    # Process every file with the model loaded once. A failure on one file
    # (missing, too long, or a runtime error) is isolated so the batch keeps going.
    total = len(args.srt_files)
    succeeded = 0
    failed = []
    for i, srt_file in enumerate(args.srt_files, start=1):
        print("\n" + "#" * 60)
        print(f"# [{i}/{total}] {srt_file}")
        print("#" * 60)

        if not os.path.exists(srt_file):
            print(f"[-] File not found, skipping: {srt_file}")
            failed.append(srt_file)
            continue

        try:
            if generate_notes(srt_file, model, tokenizer, max_seq_length, force=args.force):
                succeeded += 1
            else:
                failed.append(srt_file)
        except Exception as e:
            print(f"[-] Error processing {srt_file}: {e}")
            failed.append(srt_file)

    print("\n" + "=" * 60)
    print(f"Done: {succeeded}/{total} file(s) processed successfully.")
    if failed:
        print(f"Failed/skipped ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
