import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os

class TextDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据文件
        with open(os.path.join(data_dir, 'X.txt'), 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
        with open(os.path.join(data_dir, 'Y.txt'), 'r', encoding='utf-8') as f:
            self.labels = f.readlines()
        with open(os.path.join(data_dir, 'YL1.txt'), 'r', encoding='utf-8') as f:
            self.labels_level1 = f.readlines()
        with open(os.path.join(data_dir, 'YL2.txt'), 'r', encoding='utf-8') as f:
            self.labels_level2 = f.readlines()

        assert len(self.texts) == len(self.labels) == len(self.labels_level1) == len(self.labels_level2), "数据文件行数不匹配"

        # 构建标签到ID的映射
        self.label_set = sorted(set([label.strip() for label in self.labels]))
        self.label2id = {label: idx for idx, label in enumerate(self.label_set)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].strip()
        label = self.labels[idx].strip()
        label_level1 = self.labels_level1[idx].strip()
        label_level2 = self.labels_level2[idx].strip()

        # 创建层次结构提示符，将标签提示符附加到文本之后
        hierarchical_prompt = f"{text} [SEP] Level1: {label_level1} [SEP] Level2: {label_level2}"

        inputs = self.tokenizer.encode_plus(
            hierarchical_prompt,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in inputs.items()}

        # 将标签转换为张量（假设是单标签分类）
        label_id = self.label2id[label]
        item['labels'] = torch.tensor(label_id, dtype=torch.long)

        return item

def create_dataloader(data_dir, tokenizer, max_length, batch_size, shuffle=True):
    dataset = TextDataset(data_dir, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader