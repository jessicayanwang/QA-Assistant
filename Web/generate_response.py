import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import pipeline, BigBirdForQuestionAnswering, BigBirdTokenizer, AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import openai
import re
import time
import pickle
from nltk.tokenize import sent_tokenize
import asyncio

final_response_started = False
openai.api_key = "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"

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

async def get_interim_prompt(question):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": "You are a presenter at a conference answering audience questions."},
                {"role": "user", "content": question}, 
                {"role": "system", "content": "What should I say to keep the conversation going while I think \
                 of an answer to the given question? Consider techniques such as repeating the question, asking for clarification, etc \
                 Produce a response that is about 15 words long."}
        ], 
        stream=True
    )
    for chunk in response:
        if chunk['choices'][0]["finish_reason"] != "stop":
            yield chunk['choices'][0]['delta']['content']

def get_response(question, prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": "You are a presenter at a conference answering audience questions."},
                {"role": "user", "content": question}, 
                {"role": "system", "content": prompt}
        ], 
    )
    return response["choices"][0]["message"]["content"]

def generate_answer(question, context, max_qa_length=4096):
    qa_model = BigBirdForQuestionAnswering.from_pretrained('jyw22/qa_model')
    qa_tokenizer = BigBirdTokenizer.from_pretrained('jyw22/qa_tokenizer')
    qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)
    
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
        # Get model prediction
        pred = qa_pipeline(question=question, context=chunk)
        answers.append((pred["answer"], pred["score"]))

    for ans, conf in answers:
        if conf > max_confidence:
            max_confidence = conf
            best_answer = ans

    print('answer:', best_answer)
    return best_answer, max_confidence

def generate_fact(question, context, max_similarity_length=512):
    similarity_model = AutoModel.from_pretrained('jyw22/sentence_similarity')
    similarity_tokenizer = AutoTokenizer.from_pretrained('jyw22/sentence_similarity')

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
    top_fact = None
    top_similarity = -1
    for chunk in context_chunks:
        # Process each sentence in the chunk
        sentences = sent_tokenize(chunk)
        for sentence in sentences:
            inputs2 = similarity_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            embeddings2 = similarity_model(**inputs2)[0].mean(dim=1).reshape(-1)
            similarity = 1 - cosine(embeddings1.detach().numpy(), embeddings2.detach().numpy())
            # Update top fact and similarity if current sentence is more similar
            if similarity > top_similarity:
                top_fact = sentence
                top_similarity = similarity

    print(f"fact: {top_fact}")
    return top_fact

async def generate_response(question, max_qa_length=4096, max_similarity_length=512):
    global final_response_started
    starttime = time.time()
    
    # Load context
    with open("context.pk1", 'rb') as file:
        context = pickle.load(file)

    # Generate outputs
    answer, confidence = generate_answer(question, context, max_qa_length)
    fact = generate_fact(question, context, max_similarity_length)
    print('Models took {} seconds'.format(time.time() - starttime))
    prompt = f"You want to respond to a question." \
             f"You must mention this answer: {answer}." \
             f"Include the following additional fact if you find it relevant to the question: {fact}." \
             f"Please generate a smooth script to answer this question as requested." \
             f"Produce a response that is no longer than 30 words." \

    if confidence < 0.4:
        confidence_str = 'low'
    elif (confidence >= 0.4 and confidence < 0.7):
        confidence_str = 'mid'
    else:
        confidence_str = 'high'

    final_response_started = True

    return get_response(question, prompt), confidence_str

async def main(question):
    global final_response_started
    final_response_started = False

    # Start generating the final response
    response_task = asyncio.create_task(generate_response(question))

    # Start streaming interim prompts until the final response has started
    async for prompt in get_interim_prompt(question):
        if final_response_started: 
            break
        yield prompt, 'high'  # Yield interim prompts with high confidence

    # Now, yield chunks from the final response
    response, confidence_str = await(response_task)
    yield "\n" + response, confidence_str