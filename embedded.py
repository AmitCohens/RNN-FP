import json

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

def embedDescriptions():
    with open('data.json', 'r') as f:
        movies = json.load(f)

    # Load the pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    for movie in movies:
        description = movies[movie]['description']
        # Tokenize the input
        inputs = tokenizer(description, return_tensors='pt', padding=True, truncation=True)

        # Pass the input through the model and get the output
        outputs = model(**inputs)

        # Get the last hidden state of the model output
        last_hidden_state = outputs.last_hidden_state

        # Get the mean of the last hidden state along the second dimension
        mean_last_hidden_state = torch.mean(last_hidden_state, dim=1)

        # Convert the tensor to a numpy array
        embedding_vector = mean_last_hidden_state.detach().numpy()

        # Assign the embedding vector to a new key in the dictionary
        movies[movie]['embedded'] = embedding_vector[0].tolist()

        print(f"Movie: {movies[movie]['name']}\t{embedding_vector}")


    with open('data.json', 'w') as f:
        json.dump(movies, f, indent=4)


if __name__ == '__main__':
    embedDescriptions()