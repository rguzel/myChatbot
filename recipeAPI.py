import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import os
import joblib

app = Flask(__name__)

# Load the embeddings and the model
embeddings = np.load('email_embeddings.npy')
model = SentenceTransformer('all-MiniLM-L6-v2')

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

def find_similar_recipes(query, embeddings, recipes, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    scores = util.semantic_search(query_embedding, embeddings, top_k=top_k)
    return scores[0]  # Return the top_k results

@app.route('/find_similar', methods=['POST'])
def find_similar():
    data = request.get_json()
    query = data['query']
    top_k = data.get('top_k', 5)  # Default top_k to 5 if not provided
    
    # Find similar recipes
    results = find_similar_recipes(query, embeddings, recipes, top_k)
    
    # Fetch the details of similar recipes from the DataFrame
    response_data = []
    for result in results:
        index = result['corpus_id']
        score = result['score']
        recipe_details = recipes.iloc[index].to_dict()
        recipe_details['similarity_score'] = score
        response_data.append(recipe_details)
    
    # Prepare the response
    response = {
        'query': query,
        'results': response_data
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)  # Run the server
