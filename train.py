import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import cfg
from src.model import PlagiarismModel
from src.data_utils import load_dataset_from_dir

class PlagiarismDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        anchor = sample["anchor"]
        sentence_0 = sample["0"]  
        sentence_1 = sample["1"]  

        anchor_enc = self.tokenizer(anchor, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        sentence_0_enc = self.tokenizer(sentence_0, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        sentence_1_enc = self.tokenizer(sentence_1, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "anchor": anchor_enc,
            "sentence_0": sentence_0_enc,
            "sentence_1": sentence_1_enc
        }

# 训练函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()  
    print("Training started!")  
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):

        anchor = {key: val.squeeze(0).to(device) for key, val in batch["anchor"].items()}
        sentence_0 = {key: val.squeeze(0).to(device) for key, val in batch["sentence_0"].items()}
        sentence_1 = {key: val.squeeze(0).to(device) for key, val in batch["sentence_1"].items()}

        similarity_0 = model(anchor, sentence_0)
        similarity_1 = model(anchor, sentence_1)

        labels_0 = torch.zeros_like(similarity_0)
        labels_1 = torch.ones_like(similarity_1)
        loss_0 = criterion(similarity_0, labels_0)
        loss_1 = criterion(similarity_1, labels_1)
        loss = (loss_0 + loss_1) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Training finished for this epoch!")  
    return total_loss / len(dataloader)

if __name__ == "__main__":
    args = cfg.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading dataset...")
    data = load_dataset_from_dir(args.article_directory)  
    print(f"Loaded {len(data)} samples from {args.article_directory}")

    dataset = PlagiarismDataset(data, tokenizer, args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = PlagiarismModel(args.model_name, args.dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(5):
        avg_loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
