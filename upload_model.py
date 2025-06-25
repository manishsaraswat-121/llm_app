from huggingface_hub import HfApi, upload_folder
from huggingface_hub import login

login()  # enter your HF token

repo_id = "maneesh20022002/tiny-gpt2-medquad"
api = HfApi()
api.create_repo(repo_id, private=False)

upload_folder(
    folder_path="C:/Users/manis/Downloads/medical-llm-project/models/tiny-gpt2-medquad",
    repo_id=repo_id,
    path_in_repo=".",
    repo_type="model"
)
print("✅ Model uploaded to:", repo_id)
