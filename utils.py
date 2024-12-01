import torch

import numpy as np
import os


def encode_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, return_dict=True)
    sentence_embedding = outputs.last_hidden_state.mean(dim=1)
    return sentence_embedding


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def load_fvecs(file_path):
    with open(file_path, 'rb') as f:
        data = []
        while True:
            dim_data = f.read(4)
            if not dim_data:
                break
            dim = np.frombuffer(dim_data, dtype='int32')[0]
            vec = np.frombuffer(f.read(dim * 4), dtype='float32')
            data.append(vec)
        if data == []:
            return data
        return np.vstack(data)


def save_fvecs(file_path, vectors):
    with open(file_path, 'wb') as f:
        for vector in vectors:
            dim = np.array([len(vector)], dtype='int32')
            dim.tofile(f)
            vector.astype('float32').tofile(f)
