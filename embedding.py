from fastembed import TextEmbedding
# This will download the model fresh
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("Model downloaded successfully!")