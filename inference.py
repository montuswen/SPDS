from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch

import os
import numpy as np
import faiss
import json
import time

import cfg
from utils import mean_pooling, load_fvecs, save_fvecs
from src.data_utils import get_articles

if __name__ == "__main__":

    args = cfg.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    articles_path = get_articles(args.article_directory)
    report_dic = {}
    for idx in range(len(articles_path)):
        report_dic[idx] = {}
        for i in range(len(articles_path)):
            report_dic[idx][i] = []
    for idx, article_path in enumerate(articles_path):
        with open(article_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            start_time = time.time()
            print(
                f'start {article_path} at time {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
            for item in data:
                sentence_id = item["sentence_id"]
                article_id = item["article_id"]
                sentence = item["sentence"]
                time1 = time.time()
                # print(f'solve sentence{sentence_id} {sentence}')

                encoded_input = tokenizer(sentence, padding=True,
                                          truncation=True, return_tensors='pt')
                with torch.no_grad():
                    model_output = model(**encoded_input)
                embedding = mean_pooling(
                    model_output, encoded_input['attention_mask'])
                embedding = F.normalize(embedding, p=2, dim=1).numpy()
                # print(f'embedding shape{embedding.shape}')
                position = [article_id, sentence_id]
                # print(f'file size {os.path.getsize(args.data_vector_path)}')
                if os.path.getsize(args.data_vector_path) > 0:
                    vector_data = load_fvecs(args.data_vector_path)
                else:
                    vector_data = np.empty((0, args.dim), dtype=np.float32)
                if os.path.getsize(args.data_index_path) > 0:
                    vector_positions = np.load(
                        args.data_index_path, allow_pickle=True).tolist()
                else:
                    vector_positions = []
                # print(f'vector_positions{vector_positions}')
                # print(f'vector_data shape {vector_data.shape}')
                if vector_data.size > 0:
                    similarity = np.dot(embedding, vector_data.T)
                    # print(f'similarity shape{similarity.shape}')
                    # print(f'similarity{similarity}')
                    indices = np.where(similarity[0] >= args.gamma)[0]
                    # print(f'indices{indices}')
                    if indices.size > 0:
                        for i in indices:
                            vector_data[i] = (
                                vector_data[i]*len(vector_positions[i]) + embedding) / (len(vector_positions[i]) + 1)
                            for position in vector_positions[i]:
                                if position[0] != article_id:
                                    report_dic[position[0]][article_id].append(
                                        [position[1], sentence_id])
                                    report_dic[article_id][position[0]].append(
                                        [sentence_id, position[1]])
                            vector_positions[i].append(position)
                    else:
                        vector_data = np.append(
                            vector_data, embedding, axis=0)
                        vector_positions.append([position])
                else:
                    vector_data = np.append(
                        vector_data, embedding, axis=0)
                    vector_positions.append([position])
                save_fvecs(args.data_vector_path, vector_data)
                np.save(args.data_index_path, np.array(
                    vector_positions, dtype=object))
            print(
                f'finish {article_path} at time {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())},use time {time.time() - start_time}')
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report_dic, f, indent=4)
