from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer

qa_model = BigBirdForQuestionAnswering.from_pretrained('jyw22/qa_model')
qa_tokenizer = BigBirdTokenizer.from_pretrained('jyw22/qa_model')

qa_model.push_to_hub("qa_model")
qa_tokenizer.push_to_hub("qa_tokenizer")

