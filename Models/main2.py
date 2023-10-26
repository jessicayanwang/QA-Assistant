import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer, AutoModel, AutoTokenizer, GPT2Tokenizer
import torch
from scipy.spatial.distance import cosine
import argparse
import PyPDF2
import re
import openai
import math


def extract_text_from_pdf(pdf_file):
    with open(pdf_file, "rb") as file:
        # Initialize a PdfReader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Initialize a string to store the extracted text
        extracted_text = ""

        # Loop through all pages in the PDF file and extract text
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()

    # Save the extracted text to a text file
    with open("context.txt", "w", encoding="utf-8") as output_file:
        output_file.write(extracted_text)


def trim_to_last_complete_sentence(text):
    # Split the text into individual sentences
    sentences = re.split(r'(?<=[\.\?!])\s+', text)
    if len(sentences) > 1:
        last_sentence = sentences[-1].strip()
        if not last_sentence.endswith((".", "?", "!")):
            # Last sentence is incomplete, remove it
            sentences = sentences[:-1]
    # Join the complete sentences and return
    return " ".join(sentences)


def get_response(prompt):
    openai.api_key = "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"

    # Load GPT-3.5 tokenizer
    gpt3_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Tokenize the prompt
    input_tokens = gpt3_tokenizer.encode(prompt, return_tensors="pt")
    input_length = input_tokens.shape[1]
    # Token length is greater than the max length gpt-3.5 can take in
    if input_length > 4096:
        # Split the context into chunks based on token length
        num_chunk = math.ceil((4096 + input_length) / 4096)
        # Calculate the approximate chunk size
        chunk_size = input_length // num_chunk

        text_chunks = []
        for i in range(num_chunk):
            chunk = gpt3_tokenizer.decode(input_tokens[0][i:i + chunk_size])
            text_chunks.append(chunk)

        # Process each chunk and concatenate the results
        response_text = ""
        for chunk in text_chunks:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": chunk
                    }
                ]
            )
            generated_text = response.choices[0]['message']['content'].strip()
            response_text += trim_to_last_complete_sentence(generated_text)
        return response_text
    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        generated_text = response.choices[0]['message']['content'].strip()

    return trim_to_last_complete_sentence(generated_text)


def generate_response(context, question, max_length=4096):
    sentences = context.split(".")

    # Load the fine-tuned QA model and tokenizer
    qa_model = BigBirdForQuestionAnswering.from_pretrained('qa_model')
    qa_tokenizer = BigBirdTokenizer.from_pretrained('qa_model')

    # Tokenize the input
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt")

    # Check if the input exceeds maximum length
    input_length = inputs.input_ids.shape[-1]
    if input_length > max_length:
        prompt = f"Please summarize the following context while retaining as much important information as possible: {context}"
        context = get_response(prompt)
        inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt")

    # Get model predictions
    output = qa_model(**inputs)
    # Extract answer from the output
    start = torch.argmax(output.start_logits, dim=1).item()
    end = torch.argmax(output.end_logits, dim=1).item()
    answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end + 1]))
    print("Answer:", answer)

    # Load the fine-tuned sentence similarity model and tokenizer
    similarity_model = AutoModel.from_pretrained('similarity_model')
    similarity_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')

    inputs1 = similarity_tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    embeddings1 = similarity_model(**inputs1)[0].mean(dim=1).reshape(-1)

    # Split the context into chunks with a maximum token length of 128
    context_tokens = similarity_tokenizer.encode(context)
    similarity_chunk_size = 128 - len(inputs1["input_ids"][0]) - 2  # Subtracting question length and special tokens

    # Check if the context is less than 128 tokens
    if len(context_tokens) < 128:
        context_chunks = [context]
    else:
        context_chunks = []
        for i in range(0, len(context_tokens), similarity_chunk_size):
            chunk = similarity_tokenizer.decode(context_tokens[i:i + similarity_chunk_size])
            context_chunks.append(chunk)

    # Process each chunk and calculate similarities
    similarities = []
    for chunk in context_chunks:
        inputs2 = similarity_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        embeddings2 = similarity_model(**inputs2)[0].mean(dim=1).reshape(-1)
        similarity = 1 - cosine(embeddings1.detach().numpy(), embeddings2.detach().numpy())
        similarities.append((chunk, similarity))

    # Find the top 2 most similar chunks
    top1_index = 0
    top2_index = 1
    if similarities[top1_index][1] < similarities[top2_index][1]:
        top1_index, top2_index = top2_index, top1_index

    for i in range(2, len(similarities)):
        if similarities[i][1] > similarities[top1_index][1]:
            top2_index = top1_index
            top1_index = i
        elif similarities[i][1] > similarities[top2_index][1]:
            top2_index = i

    fact1, _ = similarities[top1_index]
    fact2, _ = similarities[top2_index]
    print("fact1:", fact1)
    print("fact2:", fact2)

    prompt = f"I am presenting at a conference and I have got this question: " \
             f"{question}. I know that answers are {answer}. I want to respond to the question " \
             f"while mentioning the answers and incorporating these relevant sentences " \
             f"{fact1}, {fact2}. Please generate a script that I could read for this question. " \
             f"Begin response with sentences like 'Just to repeat, you are saying that', 'That's a good question' or " \
             f"'Interesting question'. Produce answer that is about 50 words long." \

    response = get_response(prompt)
    print("Final response:", response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pass context file name and question as arguments.")
    parser.add_argument("--context", type=str, required=True, help="The PDF file containing the context (without the '.pdf' extension).")
    parser.add_argument("--question", type=str, required=True, help="The question to be answered.")
    args = parser.parse_args()
    # Convert pdf file to a text file
    pdf_file_path = args.context + '.pdf'
    extract_text_from_pdf(pdf_file_path)
    with open('context.txt', 'r') as f:
        context = f.read()

    question = args.question

    generate_response(context, question)

