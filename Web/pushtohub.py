from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
from huggingface_hub import HfApi

# Upload qa_model
# qa_model = BigBirdForQuestionAnswering.from_pretrained('jyw22/qa_model')
# qa_tokenizer = BigBirdTokenizer.from_pretrained('jyw22/qa_model')

# qa_model.push_to_hub("qa_model")
# qa_tokenizer.push_to_hub("qa_tokenizer")

# Upload similarity_model
api = HfApi()
api.upload_folder(
    folder_path="Web/similarity_model",
    repo_id="jyw22/sentence_similarity",
    repo_type="model",
)

