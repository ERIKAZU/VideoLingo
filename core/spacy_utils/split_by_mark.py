import os
import pandas as pd
import warnings
from core.spacy_utils.load_nlp_model import init_nlp, SPLIT_BY_MARK_FILE
from core.utils.config_utils import load_key, get_joiner
from rich import print as rprint

warnings.filterwarnings("ignore", category=FutureWarning)

def split_by_mark(nlp):
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language # consider force english case
    joiner = get_joiner(language)
    rprint(f"[blue]🔍 Using {language} language joiner: '{joiner}'[/blue]")
    chunks = pd.read_excel("output/log/cleaned_chunks.xlsx")
    chunks.text = chunks.text.apply(lambda x: x.strip('"').strip(""))
    
    # join with joiner
    input_text = joiner.join(chunks.text.to_list())

    # spaCy has a max input size (~49K bytes). Process in batches if needed.
    MAX_BYTES = 40000  # leave buffer below spaCy's 49149 byte limit
    if len(input_text.encode('utf-8')) > MAX_BYTES:
        rprint(f"[yellow]⚠️ Input text ({len(input_text.encode('utf-8'))} bytes) exceeds spaCy limit, splitting into batches...[/yellow]")
        # Split at sentence-ending punctuation marks to keep natural boundaries
        import re
        # Split on common sentence-ending punctuation for CJK and Western languages
        parts = re.split(r'(?<=[。！？\.\!\?])', input_text)
        batches = []
        current_batch = ""
        for part in parts:
            if len((current_batch + part).encode('utf-8')) > MAX_BYTES and current_batch:
                batches.append(current_batch)
                current_batch = part
            else:
                current_batch += part
        if current_batch:
            batches.append(current_batch)
        rprint(f"[blue]📦 Split into {len(batches)} batches for spaCy processing[/blue]")
        
        # Process each batch through spaCy and collect sentences
        all_sents = []
        for batch in batches:
            doc = nlp(batch)
            assert doc.has_annotation("SENT_START")
            all_sents.extend(list(doc.sents))
    else:
        doc = nlp(input_text)
        assert doc.has_annotation("SENT_START")
        all_sents = list(doc.sents)

    # skip - and ...
    sentences_by_mark = []
    current_sentence = []
    
    # iterate all sentences
    for sent in all_sents:
        text = sent.text.strip()
        
        # check if the current sentence ends with - or ...
        if current_sentence and (
            text.startswith('-') or 
            text.startswith('...') or
            current_sentence[-1].endswith('-') or
            current_sentence[-1].endswith('...')
        ):
            current_sentence.append(text)
        else:
            if current_sentence:
                sentences_by_mark.append(' '.join(current_sentence))
                current_sentence = []
            current_sentence.append(text)
    
    # add the last sentence
    if current_sentence:
        sentences_by_mark.append(' '.join(current_sentence))

    with open(SPLIT_BY_MARK_FILE, "w", encoding="utf-8") as output_file:
        for i, sentence in enumerate(sentences_by_mark):
            if i > 0 and sentence.strip() in [',', '.', '，', '。', '？', '！']:
                # ! If the current line contains only punctuation, merge it with the previous line, this happens in Chinese, Japanese, etc.
                output_file.seek(output_file.tell() - 1, os.SEEK_SET)  # Move to the end of the previous line
                output_file.write(sentence)  # Add the punctuation
            else:
                output_file.write(sentence + "\n")
    
    rprint(f"[green]💾 Sentences split by punctuation marks saved to →  `{SPLIT_BY_MARK_FILE}`[/green]")

if __name__ == "__main__":
    nlp = init_nlp()
    split_by_mark(nlp)
