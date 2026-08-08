import sys
import os
import json
import argparse
from unsloth import FastLanguageModel

DEFAULT_MAX_SEQ_LENGTH = 16384
MAX_NEW_TOKENS = 2000

def generate_notes(srt_file, model, tokenizer, max_seq_length):
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
        sys.exit(1)
        
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
    output_md = srt_file.replace(".srt", "_notes.md")
    if output_md == srt_file:
        output_md += "_notes.md"
        
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n[+] Notes saved to: {output_md}")
    
    # Extract the TSV block and save to files
    tsv_match = re.search(r'```tsv\n(.*?)```', result, re.DOTALL | re.IGNORECASE)
    if tsv_match:
        tsv_content = tsv_match.group(1).strip()
        
        output_tsv = srt_file.replace(".srt", ".tsv")
        if output_tsv == srt_file: output_tsv += ".tsv"
        with open(output_tsv, "w", encoding="utf-8") as f:
            f.write(tsv_content)
        print(f"[+] TSV table saved to: {output_tsv}")
        
        try:
            import genanki
            deck_name = os.path.basename(srt_file).replace(".srt", "")
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
                    
            output_apkg = srt_file.replace(".srt", ".apkg")
            if output_apkg == srt_file: output_apkg += ".apkg"
            genanki.Package(anki_deck).write_to_file(output_apkg)
            print(f"[+] Anki deck saved to: {output_apkg}")
        except ImportError:
            print("[-] 'genanki' library is not installed. APKG file was not generated.")
    else:
        print("[-] TSV block not found in the generated response.")

def main():
    parser = argparse.ArgumentParser(description="SRT to Markdown & Anki")
    parser.add_argument("srt_file", help="Path to the .srt file")
    parser.add_argument("--context", type=int, default=None, help="Override context length (max_seq_length)")
    args = parser.parse_args()
    
    srt_file = args.srt_file
    if not os.path.exists(srt_file):
        print(f"File not found: {srt_file}")
        sys.exit(1)
        
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

    # Override with command line argument if provided
    if args.context is not None:
        max_seq_length = args.context

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
    
    generate_notes(srt_file, model, tokenizer, max_seq_length)

if __name__ == "__main__":
    main()
