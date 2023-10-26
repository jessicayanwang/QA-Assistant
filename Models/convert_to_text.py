

import re


def trim_to_complete_sentences(text):
    # Split the text into individual sentences
    sentences = re.split(r'(?<=[\.\?!])\s+', text)
    if len(sentences) > 1:
        last_sentence = sentences[-1].strip()
        if not last_sentence.endswith((".", "?", "!")):
            # Last sentence is incomplete, remove it
            sentences = sentences[:-1]
    # Join the complete sentences and return
    return " ".join(sentences)




if __name__ == '__main__':
    text = "Hi, my name is Jessica. What is your name"
    trimmed_text = trim_to_complete_sentences(text)
    print(trimmed_text)
