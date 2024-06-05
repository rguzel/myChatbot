
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import joblib
import os

# Load your CSV file
cache_path = 'C:\\Data\\cache\\recipes_data.pkl'


def load_data(csv_path, cache_path):
    # Ensure the cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        # Try to load from cache
        data = joblib.load(cache_path)
        print("Loaded data from cache.")
    except FileNotFoundError:
        # If not cached, load from CSV and cache it
        data = pd.read_csv(csv_path)
        joblib.dump(data, cache_path)
        print("Loaded data from CSV and cached it.")
    return data

# Load your data using caching
recipes = load_data('C:\\Data\\recipes_data.csv\\recipes_data.csv', cache_path)


# Load the embeddings from the file
embeddings = np.load('email_embeddings.npy')
model = SentenceTransformer('all-MiniLM-L6-v2')

def find_similar_recipes(query, embeddings, recipes, top_k=5):
    query_embedding = model.encode([query])
    scores = util.semantic_search(query_embedding, embeddings, top_k=top_k)
    return [(recipes.iloc[score['corpus_id']]['title'], score['score']) for score in scores[0]]


def find_similar_emails(query_embedding, embeddings, top_k=5):
    # Compute cosine similarities between the query and all embeddings
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)
    
    # Get the top k most similar indices
    top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
    
    return top_k_indices, similarities[0][top_k_indices]

# Example: Assuming 'model' is your SentenceTransformer model
def get_query_embedding(query, model):
    return model.encode(query)

# Example usage
query = "How do I make a tomatoe sauce?"
#similar_recipes = find_similar_recipes(query, embeddings, recipes)
query_embedding = get_query_embedding(query, model)

# Find similar emails (or recipes)
indices, scores = find_similar_emails(query_embedding, embeddings)

#print(similar_recipes)

# Print the most similar emails or recipes
for index in indices:
    print(recipes.iloc[index])