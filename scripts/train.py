import torch
import torch.nn as nn
from utils.data_loader import create_dataloader
from utils.model_utils import HierarchicalTextModel
from transformers import BertTokenizer
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.nn.functional as F
import networkx as nx
import os
from tqdm import tqdm

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size).to(z.device)
    sim_matrix = sim_matrix - mask * 1e12  # 防止自身匹配

    positives = torch.cat([torch.diag(sim_matrix, batch_size), torch.diag(sim_matrix, -batch_size)], dim=0)
    negatives = sim_matrix[mask == 0].view(2 * batch_size, -1)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2 * batch_size).long().to(z.device)

    loss = F.cross_entropy(logits, labels)
    return loss


def train():
    # 加载配置
    with open('D:/Project/tectClass/configs/config.yaml', 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 初始化
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])
    train_dataloader = create_dataloader(config['data']['train_dir'], tokenizer, config['model']['max_length'],
                                         config['training']['batch_size'], shuffle=True)
    label_graph = nx.DiGraph()  # 构建标签层次图，这需要根据您的标签层次结构进行实现
    # TODO: 添加节点和边到label_graph
    ############
    with open(os.path.join(config['data']['train_dir'], 'YL1.txt'), 'r', encoding='utf-8') as f:
        level1_labels = [line.strip() for line in f.readlines()]
    with open(os.path.join(config['data']['train_dir'], 'YL2.txt'), 'r', encoding='utf-8') as f:
        level2_labels = [line.strip() for line in f.readlines()]
    unique_level1_labels = set(level1_labels)
    unique_level2_labels = set(level2_labels)
    for label in unique_level1_labels:
        label_graph.add_node(label)
    for parent_label, child_label in zip(level1_labels, level2_labels):
        label_graph.add_node(child_label)
        label_graph.add_edge(parent_label, child_label)
    for label in label_graph.nodes():
        label_graph.nodes[label]['name'] = label

    ############

    model = HierarchicalTextModel(config['model']['bert_model_name'], config['model']['num_labels'],
                                  config['model']['label_embedding_dim'], label_graph).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    writer = SummaryWriter(log_dir='logs/experiment')
    temperature = config['training']['temperature']

    # 训练循环
    model.train()
    for epoch in range(config['training']['num_epochs']):
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 正样本生成（根据标签的重要性采样tokens）
            positive_input_ids, positive_attention_mask = generate_positive_samples(batch, tokenizer,
                                                                                    config['model']['max_length'])

            positive_input_ids = positive_input_ids.to(device)
            positive_attention_mask = positive_attention_mask.to(device)

            logits, z_i, z_j = model(input_ids, attention_mask, positive_input_ids, positive_attention_mask)

            # 分类损失
            classification_loss = criterion(logits, labels)

            # 对比学习损失
            contrastive_loss = nt_xent_loss(z_i, z_j, temperature)

            # 总损失
            loss = classification_loss + contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_dataloader):.4f}")
        avg_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f'Epoch {epoch + 1}/{config["training"]["num_epochs"]}, Loss: {avg_loss}')

    # 保存模型
    torch.save(model.state_dict(), 'D:/Project/tectClass/models/hierarchical_model_2.pth')
    writer.close()


def generate_positive_samples(batch, tokenizer, max_length):
    # 根据标签的重要性采样tokens，生成正样本
    # 这里提供一个简单的实现，实际根据需求进行调整
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    positive_input_ids = input_ids.clone()
    positive_attention_mask = attention_mask.clone()

    # 随机mask一部分tokens
    mask_ratio = 0.2
    for i in range(input_ids.size(0)):
        input_length = attention_mask[i].sum().item()
        num_mask = int(input_length * mask_ratio)
        mask_indices = torch.randperm(int(input_length))[:num_mask]
        positive_input_ids[i, mask_indices] = tokenizer.mask_token_id

    return positive_input_ids, positive_attention_mask


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train()