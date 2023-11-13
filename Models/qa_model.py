import torch
import huggingface as hf
from datasets import load_dataset
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import BigBirdTokenizerFast, BigBirdForQuestionAnswering, AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint


PUNCTUATION_SET_TO_EXCLUDE = set(''.join(['‘', '’', '´', '`', '.', ',', '-', '"']))


def format_dataset(example):
    # the context might be comprised of multiple contexts => me merge them here
    example["context"] = example["context"]
    example["targets"] = example["answers"]["text"]
    example["sentences"] = example["context"].split(".")
    example["start"] = example["answers"]["answer_start"][0]
    example["answers"] = example["answers"]["text"][0]
    return example


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))


def get_sub_answers(answers, begin=0, end=None):
    return [" ".join(x.split(" ")[begin:end]) for x in answers if len(x.split(" ")) > 1]


def expand_to_aliases(given_answers, make_sub_answers=False):
    if make_sub_answers:
    # if answers are longer than one word, make sure a predictions is correct if it corresponds to the complete 1: or :-1 sub word
    # *e.g.* if the correct answer contains a prefix such as "the", or "a"
        given_answers = given_answers + get_sub_answers(given_answers, begin=1) + get_sub_answers(given_answers, end=-1)
    answers = []
    for answer in given_answers:
        alias = answer.replace('_', ' ').lower()
        alias = ''.join(c if c not in PUNCTUATION_SET_TO_EXCLUDE else ' ' for c in alias)
        answers.append(' '.join(alias.split()).strip())
    return set(answers)


def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = torch.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]


def prepare_training_data(example, tokenizer):
    encoded_inputs = tokenizer(example['question'], example['context'], padding = 'max_length', truncation=True, max_length = 4096)
    start_char = example['start']
    answer = example['answers']
    end_char = start_char + len(answer)
    start = encoded_inputs.char_to_token(start_char, sequence_index=1)
    end = encoded_inputs.char_to_token(end_char, sequence_index=1)
    example["start"] = start
    example["end"] = end
    return example


def my_collate_fn(examples, tokenizer):
    encoded_inputs = tokenizer([example['question'] for example in examples],
                               [example['context'] for example in examples],
                               return_tensors='pt', padding='max_length', truncation=True, max_length=4096)
    start_positions = torch.tensor([example['start'] for example in examples])
    end_positions = torch.tensor([example['end'] for example in examples])
    return {'input_ids': encoded_inputs['input_ids'],
            'attention_mask': encoded_inputs['attention_mask'],
            'start_positions': start_positions,
            'end_positions': end_positions}


def train(model, tokenizer, optimizer: AdamW, train_set, validation_set, num_train_epochs: int, batch_size: int,
          max_input_length: int = 4096, gradient_accumulation_steps: int = 1):
    my_trainset_dataloader = DataLoader(train_set, batch_size=batch_size, collate_fn=lambda batch: my_collate_fn(batch, tokenizer))

    # set training mode on the model
    model.train()
    model.to('cuda')

    best_em = 0  # initialize the best exact match score
    best_model_state = None

    optimizer.param_groups[0]['lr'] /= gradient_accumulation_steps

    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.
        step_count = 0

        for batch in tqdm(my_trainset_dataloader):
            input_ids = batch["input_ids"].to('cuda')
            attention_mask = batch["attention_mask"].to('cuda')
            start_positions = batch["start_positions"].to('cuda')
            end_positions = batch["end_positions"].to('cuda')

            # Apply gradient checkpoint
            def custom_forward(inputs):
                input_ids, attention_mask, start_positions, end_positions = inputs
                return model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions,
                             end_positions=end_positions)

            inputs = (input_ids, attention_mask, start_positions, end_positions)
            outputs = checkpoint(custom_forward, inputs, use_reentrant=False)
            loss = outputs.loss

            # Backward pass with gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            epoch_train_loss += loss.item() * batch_size

            step_count += 1
            if step_count % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            input_ids = batch["input_ids"].to('cpu')
            attention_mask = batch["attention_mask"].to('cpu')
            start_positions = batch["start_positions"].to('cpu')
            end_positions = batch["end_positions"].to('cpu')
            torch.cuda.empty_cache()

        print(f"epoch={epoch + 1}/{num_train_epochs}")
        print(f"\t Train loss = {epoch_train_loss / len(train_set):.4f}")

        model.eval()
        results = validation_set.map(lambda x: evaluate(x, model=model, tokenizer=tokenizer))
        em = 100 * sum(results['match']) / len(results)
        print("Exact Match (EM): {:.2f}".format(em))

        if em > best_em:
            best_em = em
            best_model_state = model.state_dict()

    # Save the best model state using model.save_pretrained()
    model.load_state_dict(best_model_state)
    model.save_pretrained('qa_model')
    tokenizer.save_pretrained('qa_model')



def evaluate(example, model, tokenizer):
    # encode question and context so that they are seperated by a tokenizer.sep_token and cut at max_length
    encoding = tokenizer(example["question"], example["context"], return_tensors="pt", max_length=4096, padding="max_length", truncation=True)
    input_ids = encoding.input_ids.to("cuda")

    with torch.no_grad():
        start_scores, end_scores = model(input_ids=input_ids).to_tuple()

    start_score, end_score = get_best_valid_start_end_idx(start_scores[0], end_scores[0], top_k=8, max_size=16)

    # convert the input ids back to actual tokens
    all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
    answer_tokens = all_tokens[start_score: end_score + 1]
    example["output"] = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    #.replace('"', '')  # remove space prepending space token and remove unnecessary '"'

    answers = expand_to_aliases(example["targets"], make_sub_answers=True)
    predictions = expand_to_aliases([example["output"]])

    # if there is a common element, it's a match
    example["match"] = len(list(answers & predictions)) > 0

    return example



if __name__ == '__main__':
    # load data
    train_raw = load_dataset("adversarial_qa", "adversarialQA", split="train[:50]")
    validation_raw = load_dataset("adversarial_qa", "adversarialQA", split="validation[:10]")
    # format validation + train dataset
    validation_dataset = validation_raw.map(format_dataset, remove_columns=["title", "metadata"])
    train_dataset = train_raw.map(format_dataset, remove_columns=["title", "metadata"])
    # only include samples with context
    train_dataset = train_dataset.filter(lambda x: len(x["context"]) > 0)
    validation_dataset = validation_dataset.filter(lambda x: len(x["context"]) > 0)
    short_validation_dataset = validation_dataset.filter(lambda x: (len(x['question']) + len(x['context'])) < 1000)  # <-200 is the max length to include in the set. It needs to be changed to incorporate richer data points.

    # load bigbird tokenizer and model
    tokenizer = BigBirdTokenizerFast.from_pretrained("google/bigbird-base-trivia-itc")
    model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-base-trivia-itc")

    # prepare training data for training
    train_prepared = train_dataset.map(lambda x: prepare_training_data(x, tokenizer=tokenizer), remove_columns=['answers'])

    # start training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train(model=model, tokenizer=tokenizer, optimizer=optimizer, train_set=train_prepared,
        validation_set=validation_dataset, num_train_epochs=3, batch_size=4, gradient_accumulation_steps=3)