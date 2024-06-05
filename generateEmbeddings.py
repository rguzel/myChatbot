import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util



# print("Is CUDA available: ", torch.cuda.is_available())
# print("CUDA version: ", torch.version.cuda)
# if torch.cuda.is_available():
#     print("CUDA Device Name: ", torch.cuda.get_device_name(0))


model = SentenceTransformer('all-MiniLM-L6-v2')


# To check if CUDA is available in PyTorch (which sentence transformers use)

# if torch.cuda.is_available():
#     print("CUDA is available. Using GPU.")
#     model = model.to(torch.device("cuda"))
# else:
#     print("CUDA not available. Using CPU.")


# Load your CSV file
recipes = pd.read_csv('C:\\Data\\recipes_data.csv\\recipes_data.csv')

# Check the first few rows to understand its structure
#print(recipes.head())


# Select only the first 100 rows
#subset_data = recipes.head(100)

# Convert recipe instructions to embeddings
# Generate embeddings
recipe_embeddings = model.encode(recipes['directions'].tolist(), show_progress_bar=True)
#recipe_embeddings = model.encode(recipes['directions'].tolist(), show_progress_bar=True)
#recipe_embeddings = model.encode(recipes['directions'].tolist())


np.save('email_embeddings.npy', recipe_embeddings)


