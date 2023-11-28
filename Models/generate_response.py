import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer, AutoModel, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
import openai
import re
import multiprocessing
import time
import pickle


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


def get_response(question, prompt):
    openai.api_key = "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"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": "You are a presenter at a conference answering audience questions."},
                {"role": "user", "content": question}, 
                {"role": "system", "content": prompt}
        ], 
        stream=True
    )
    for chunk in response:
        if chunk['choices'][0]["finish_reason"] != "stop":
            yield chunk['choices'][0]['delta']['content']

def generate_answer(question, context, return_dict, max_qa_length=4096):
    # Load the fine-tuned QA model and tokenizer
    qa_model = BigBirdForQuestionAnswering.from_pretrained('Models/qa_model')
    qa_tokenizer = BigBirdTokenizer.from_pretrained('Models/qa_model')

    # Tokenize the question separately to get the question length
    question_tokens = qa_tokenizer.encode(question, return_tensors="pt")
    question_length = question_tokens.shape[-1]

    # Tokenize the whole input
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt")

    # Check if the input exceeds maximum length
    input_length = inputs.input_ids.shape[-1]
    if input_length > max_qa_length:
        # Split the context into chunks based on token length
        context_tokens = qa_tokenizer.encode(context)
        chunk_size = max_qa_length - question_length - 50
        context_chunks = []
        for i in range(0, len(context_tokens), chunk_size):
            chunk = qa_tokenizer.decode(context_tokens[i:i + chunk_size])
            context_chunks.append(chunk)
    else:
        context_chunks = [context]

    # Process each chunk
    answers = []
    max_confidence = float('-inf')
    best_answer = None
    for chunk in context_chunks:
        # Tokenize the inputs
        inputs = qa_tokenizer.encode_plus(question, chunk, return_tensors="pt")

        # Get model predictions
        output = qa_model(**inputs)
        start = torch.argmax(output.start_logits, dim=1).item()
        end = torch.argmax(output.end_logits, dim=1).item()
        answer = qa_tokenizer.convert_tokens_to_string(
            qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end + 1]))
        confidence = output.start_logits[0, start].item() + output.end_logits[0, end].item()
        answers.append((answer, confidence))

    for ans, conf in answers:
        print(ans)
        if conf > max_confidence:
            max_confidence = conf
            best_answer = ans

    return_dict["answer"] = best_answer
    return_dict["confidence"] = max_confidence
    print('answer:', best_answer)

def generate_fact(question, context, return_dict, max_similarity_length=128):
    # Load the fine-tuned sentence similarity model and tokenizer
    similarity_model = AutoModel.from_pretrained('similarity_model')
    similarity_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')

    inputs1 = similarity_tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    embeddings1 = similarity_model(**inputs1)[0].mean(dim=1).reshape(-1)

    # Check if the context is less than 128 tokens
    context_tokens = similarity_tokenizer.encode(context)
    if len(context_tokens) < max_similarity_length:
        context_chunks = [context]
    else:
        # Split the context into chunks with a maximum token length of 128
        similarity_chunk_size = max_similarity_length - len(inputs1["input_ids"][0]) - 2  # Subtracting question length and special tokens
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

    for i in range(1, len(similarities)):
        if similarities[i][1] > similarities[top1_index][1]:
            top1_index = i

    fact1, _ = similarities[top1_index]

    return_dict["fact1"] = fact1

def generate_response(question, max_qa_length=4096, max_similarity_length=128):
    starttime = time.time()
    
    # Load context
    with open("context.pk1", 'rb') as file:
        context = pickle.load(file)

    # Generate outputs in parallel
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p1 = multiprocessing.Process(target=generate_answer, args=(question, context, return_dict, max_qa_length))
    p1.start()
    p2 = multiprocessing.Process(target=generate_fact, args=(question, context, return_dict, max_similarity_length))
    p2.start()
    p1.join()
    p2.join()
    print('Models took {} seconds'.format(time.time() - starttime))
    prompt = f"You know the answer is {return_dict['answer']}. You want to respond to the question " \
             f"while mentioning the answer and incorporating this relevant fact " \
             f"{return_dict['fact1']}. Please generate a smooth script to answer this question as requested" \
             f" Produce answer that is about 50 words long." \

    print(return_dict["confidence"])
    if return_dict["confidence"] < 0.5:
        confidence = 'low'
    elif (return_dict["confidence"] >= 0.5 and return_dict["confidence"] < 0.7):
        confidence = 'mid'
    else:
        confidence = 'high'

    for chunk in get_response(question, prompt):
        yield chunk, confidence


