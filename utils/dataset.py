import pandas as pd
import torch
import random

class Dataset:
    def __init__(self, df: pd.DataFrame):
        self.texts = []
        self.labels = []

        for idx, row in df.iterrows():
            indices = row['question_text']
            self.texts.append(indices)
            self.labels.append(row['target'])

        combined = list(zip(self.texts, self.labels))
        random.shuffle(combined)
        self.texts, self.labels = zip(*combined)

        self.texts = torch.stack([torch.tensor(indices, dtype=torch.long) for indices in self.texts])  # [num_samples, seq_len]
        self.labels = torch.tensor(self.labels, dtype=torch.float).unsqueeze(1)  # [num_samples, 1]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)