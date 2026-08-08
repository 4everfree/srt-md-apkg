import torch
from unsloth import FastLanguageModel
import sys

# Параметры
max_seq_length = 2048
dtype = None
load_in_4bit = True

import os

# Путь к папке с моделью относительно самого скрипта
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "lora_model")

print("Загрузка модели...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, # Загружаем наши обученные веса по абсолютному пути
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Оптимизация для быстрой генерации

def generate_notes(srt_text):
    system_prompt = (
        "You are a helpful assistant that converts video subtitles into detailed Markdown notes and Anki flashcards. "
        "Always follow these rules:\n"
        "1. Structure the notes using H2 (##) headers for each main topic.\n"
        "2. Anki flashcards must be formatted as a TSV block at the end, strictly using a 'Question \\t Answer' format.\n"
        "3. If the text relies heavily on visual context that is missing in the subtitles, add a warning callout like '> ⚠️ [Неясно в видео]: ...'"
    )
    prompt = f"Конвертируй следующие субтитры (SRT) в подробный Markdown конспект и сгенерируй TSV таблицу для карточек Anki.\n\nСубтитры:\n{srt_text}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    print("Генерация ответа...")
    outputs = model.generate(input_ids=inputs, max_new_tokens=2000, use_cache=True, temperature=0.3)
    
    # Декодируем только сгенерированный текст, отрезая промпт
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python inference.py <путь_к_файлу.srt>")
        sys.exit(1)
        
    srt_file = sys.argv[1]
    with open(srt_file, "r", encoding="utf-8") as f:
        srt_content = f.read()
        
    result = generate_notes(srt_content)
    print("\n" + "="*50 + "\n")
    print(result)
    
    import re
    import random
    import os
    
    # Сохраняем результат в файл Markdown
    output_md = srt_file.replace(".srt", "_notes.md")
    if output_md == srt_file:
        output_md += "_notes.md"
        
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n[+] Конспект сохранен в файл: {output_md}")
    
    # Извлекаем TSV блок и сохраняем в файлы
    tsv_match = re.search(r'```tsv\n(.*?)```', result, re.DOTALL | re.IGNORECASE)
    if tsv_match:
        tsv_content = tsv_match.group(1).strip()
        
        output_tsv = srt_file.replace(".srt", ".tsv")
        if output_tsv == srt_file: output_tsv += ".tsv"
        with open(output_tsv, "w", encoding="utf-8") as f:
            f.write(tsv_content)
        print(f"[+] TSV таблица сохранена в файл: {output_tsv}")
        
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
            print(f"[+] Колода Anki сохранена в файл: {output_apkg}")
        except ImportError:
            print("[-] Библиотека genanki не установлена. APKG файл не сгенерирован.")
    else:
        print("[-] Блок tsv не найден в сгенерированном ответе.")
