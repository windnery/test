## 二、项目目录结构

```
HierarchicalTextClassification/
├── data/                        # 用于存放数据集
│   ├── WOS/                     # 示例数据集文件夹
│       ├── train.csv            # 训练数据
│       └── eval.csv             # 评估数据
├── models/                      # 保存模型文件
├── logs/                        # 保存TensorBoard日志
├── scripts/                     # 脚本存放目录
│   ├── train.py                 # 训练脚本
│   └── evaluate.py              # 评估脚本
├── utils/                       # 辅助工具
│   ├── data_loader.py           # 数据加载模块
│   ├── model_utils.py           # 模型工具模块
│   └── visualization.py         # 可视化工具模块
├── configs/                     # 配置文件
│   └── config.yaml              # 超参数设置
└── README.md                    # 说明文档
```

### 1. 数据加载与处理 (`utils/data_loader.py`)

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os

class TextDataset(Dataset):
    """
    读取数据并处理，将 YL1 直接转换为数字。
    对于 YL2，由于每个父类中的编号均从 0 开始，
    需要对 (l1, l2) 进行组合映射，保证全局唯一，同时保存各级类别总数及父子对应关系。

    可选参数 global_label2id 和 parent_to_children 用于在构造时指定训练时计算好的映射，
    这样可以确保训练和评估时 l2 的全局编码一致。
    """
    def __init__(self, data_dir, tokenizer, max_length, global_label2id=None, parent_to_children=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 读取文本和两级标签
        with open(os.path.join(data_dir, 'X.txt'), 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f]

        with open(os.path.join(data_dir, 'YL1.txt'), 'r', encoding='utf-8') as f:
            self.level1_list = [int(line.strip()) for line in f]

        with open(os.path.join(data_dir, 'YL2.txt'), 'r', encoding='utf-8') as f:
            raw_level2 = [line.strip() for line in f]

        assert len(self.texts) == len(self.level1_list) == len(raw_level2), "文本行数与标签行数不匹配"

        # 如果提供了映射，就使用提供的 mapping，否则根据当前数据计算
        if global_label2id is None or parent_to_children is None:
            self.global_label2id = {}
            self.parent_to_children = {}
            global_l2_list = []
            for l1, l2_str in zip(self.level1_list, raw_level2):
                l2_int = int(l2_str)
                key = (l1, l2_int)
                if key not in self.global_label2id:
                    self.global_label2id[key] = len(self.global_label2id)
                global_l2_list.append(self.global_label2id[key])
                if l1 not in self.parent_to_children:
                    self.parent_to_children[l1] = set()
                self.parent_to_children[l1].add(self.global_label2id[key])
        else:
            self.global_label2id = global_label2id
            self.parent_to_children = parent_to_children
            global_l2_list = []
            for l1, l2_str in zip(self.level1_list, raw_level2):
                l2_int = int(l2_str)
                key = (l1, l2_int)
                if key not in self.global_label2id:
                    raise ValueError(f"训练时未包含的映射：{key}")
                global_l2_list.append(self.global_label2id[key])

        self.level2_list = global_l2_list

        # 计算每一级的类别数
        self.num_labels_l1 = len(set(self.level1_list))
        self.num_labels_l2 = len(self.global_label2id)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        yl1 = self.level1_list[idx]
        yl2 = self.level2_list[idx]

        # 拼接文本和标签信息，构造提示（可按需要修改）
        hierarchical_prompt = f"{text} [SEP] Level1: {yl1} [SEP] Level2: {yl2}"
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
        item['labels_l1'] = torch.tensor(yl1, dtype=torch.long)
        item['labels_l2'] = torch.tensor(yl2, dtype=torch.long)
        return item

def create_dataloader(data_dir, tokenizer, max_length, batch_size, shuffle=True, global_label2id=None,
                      parent_to_children=None):
    dataset = TextDataset(data_dir, tokenizer, max_length, global_label2id, parent_to_children)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
```

### 2. 模型定义 (`utils/model_utils.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import math
from collections import deque

def compute_shortest_distances(adj_matrix):
    """
    使用 BFS 计算所有节点对的最短距离。
    adj_matrix: [num_nodes, num_nodes], 0/1或其他值表示边。
    返回一个 [num_nodes, num_nodes] 张量，其中存储最短距离；
    不连通的部分记为 num_nodes+1。
    并保证 distances 与原先 adj_matrix 在同一 device 上。
    """
    device = adj_matrix.device
    adj_cpu = adj_matrix.detach().cpu()
    num_nodes = adj_cpu.size(0)
    distances_cpu = torch.full((num_nodes, num_nodes), float('inf'))

    for i in range(num_nodes):
        distances_cpu[i, i] = 0
        visited = [False] * num_nodes
        queue = deque([(i, 0)])
        visited[i] = True
        while queue:
            current, dist = queue.popleft()
            distances_cpu[i, current] = dist
            for neighbor in range(num_nodes):
                if adj_cpu[current, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))

    distances_cpu[distances_cpu == float('inf')] = num_nodes + 1
    distances = distances_cpu.to(device)
    return distances

class Graphormer(nn.Module):
    """
    适用于标签图的完善版 Graphormer，支持：
    1) 边缘编码 (edge encoding)
    2) 距离编码 (distance encoding)
    3) 在注意力中加入可选的空间偏置 (spatial bias)
    """
    def __init__(self, hidden_size, nhead=8, edge_types=2, max_distance=10, num_layers=2):
        """
        edge_types: 如果需要区分不同种类的边，可在初始化时指定。
        max_distance: 用于截断距离编码。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.edge_types = edge_types
        self.max_distance = max_distance
        self.num_layers = num_layers

        # (1) 边类型 embedding
        self.edge_type_embedding = nn.Embedding(self.edge_types, hidden_size)
        # (2) 距离 embedding
        self.distance_embedding = nn.Embedding(self.max_distance + 2, hidden_size)
        # (3) 可以选用用于多头空间偏置的参数，这里简化设为单一
        self.spatial_bias = nn.Parameter(torch.zeros(self.nhead, self.hidden_size))

        # 原有的 TransformerEncoder
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)

    def forward(self, label_embeddings, adjacency_matrix=None):
        # [num_nodes, hidden_size] -> [1, num_nodes, hidden_size]
        label_emb = label_embeddings.unsqueeze(0)

        if adjacency_matrix is None:
            out = self.encoder(label_emb)
            return out.squeeze(0)

        # 计算最短距离
        distances = compute_shortest_distances(adjacency_matrix)
        distances_clamped = distances.clamp(max=self.max_distance)

        # 构建 mask：对于无连接的地方可选是否屏蔽
        attention_mask = (adjacency_matrix == 0).unsqueeze(0)
        # edge_type_tensor：0~(edge_types - 1)，若大于edge_types-1则需要自行处理
        edge_type_tensor = adjacency_matrix.clone().long()
        edge_type_tensor = edge_type_tensor.clamp(0, self.edge_types - 1)

        # 边类型 embedding
        edge_type_emb = self.edge_type_embedding(edge_type_tensor)  # [num_nodes, num_nodes, hidden_size]
        # 距离 embedding
        distance_tensor = distances_clamped.long()
        distance_emb = self.distance_embedding(distance_tensor)      # [num_nodes, num_nodes, hidden_size]

        # 简单相加作为空间编码
        spatial_enc = edge_type_emb + distance_emb
        # 这里只是示例，不同头可能需要不同方式融合
        attn_spatial_bias = torch.einsum('ijh,h->ij', spatial_enc, self.spatial_bias[0])

        attn_spatial_bias_norm = attn_spatial_bias / math.sqrt(self.hidden_size)
        combined_mask = attention_mask.clone().float()
        combined_mask[0] += (-attn_spatial_bias_norm)

        # 将 mask 扩展为 [nhead, num_nodes, num_nodes] 以适配多头注意力
        combined_mask_3d = combined_mask.bool().repeat(self.nhead, 1, 1)

        out = self.encoder(label_emb, mask=combined_mask_3d)  # [1, num_nodes, hidden_size]
        return out.squeeze(0)

class HierarchicalTextModel(nn.Module):
    """
    采用 BERT 进行文本编码，结合 Graphormer 对标签图进行建模，
    最终输出 logits_l1, logits_l2。
    """
    def __init__(self, bert_model_name, num_labels_l1, num_labels_l2,
                 label_embedding_dim, label_graph, adjacency_matrix=None):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.encoder.config.hidden_size

        num_graph_nodes = len(label_graph.nodes)
        self.label_embedding = nn.Embedding(num_embeddings=num_graph_nodes,
                                            embedding_dim=label_embedding_dim)

        # 使用包含边缘编码与距离编码的 Graphormer
        self.graph_encoder = Graphormer(hidden_size=label_embedding_dim, nhead=8)

        if adjacency_matrix is not None:
            self.register_buffer("adjacency_matrix", adjacency_matrix)
        else:
            self.adjacency_matrix = None

        self.label_linear = nn.Linear(label_embedding_dim, self.hidden_size)

        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        # 两个分类器分别预测 l1 和 l2
        self.classifier_l1 = nn.Linear(self.hidden_size, num_labels_l1)
        self.classifier_l2 = nn.Linear(self.hidden_size, num_labels_l2)

    def forward(self, input_ids, attention_mask,
                positive_input_ids=None, positive_attention_mask=None):
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               output_attentions=True)
        cls_emb = outputs.last_hidden_state[:, 0, :]

        label_features = self.get_label_features()
        label_features = self.label_linear(label_features)
        combined_features = cls_emb + label_features

        logits_l1 = self.classifier_l1(combined_features)
        logits_l2 = self.classifier_l2(combined_features)

        z_i = self.projection_head(combined_features)
        z_j = None
        if positive_input_ids is not None and positive_attention_mask is not None:
            pos_outs = self.encoder(positive_input_ids,
                                    attention_mask=positive_attention_mask,
                                    output_attentions=True)
            pos_cls_emb = pos_outs.last_hidden_state[:, 0, :]
            pos_combined_features = pos_cls_emb + label_features
            z_j = self.projection_head(pos_combined_features)

        return logits_l1, logits_l2, z_i, z_j

    def get_label_features(self):
        label_embeddings = self.label_embedding.weight
        if self.adjacency_matrix is not None:
            encoded_label_features = self.graph_encoder(label_embeddings, self.adjacency_matrix)
        else:
            encoded_label_features = self.graph_encoder(label_embeddings, None)
        label_features = torch.mean(encoded_label_features, dim=0)
        return label_features
```

### 3. 训练脚本 (`scripts/train.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import networkx as nx
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from utils.data_loader import create_dataloader, TextDataset
from utils.model_utils import HierarchicalTextModel
from utils.visualization import plot_metric_curve

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    NT-Xent 损失函数，用于对比学习。
    """
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size).to(z.device)
    sim_matrix = sim_matrix - mask * 1e12

    positives = torch.cat([
        torch.diag(sim_matrix, batch_size),
        torch.diag(sim_matrix, -batch_size)
    ], dim=0)
    negatives = sim_matrix[mask == 0].view(2 * batch_size, -1)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def get_token_importance(model, input_ids, attention_mask):
    """
    计算句子中各 token 的重要度
    """
    with torch.no_grad():
        outputs = model.encoder(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_attentions=True)
        attentions = outputs.attentions
        all_attentions = torch.stack(attentions, dim=0)
        mean_attention = all_attentions.mean(dim=(0, 2))
        token_importance = mean_attention.sum(dim=-1)
    return token_importance

def generate_positive_samples(model, tokenizer, batch, threshold=0.3):
    """
    使用重要度采样替换低重要度 token 为 [MASK]，构造强化正样本。
    """
    device = next(model.parameters()).device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    importance_scores = get_token_importance(model, input_ids, attention_mask)
    min_vals = importance_scores.min(dim=1, keepdim=True)[0]
    max_vals = importance_scores.max(dim=1, keepdim=True)[0].clamp(min=1e-9)
    norm_scores = (importance_scores - min_vals) / (max_vals - min_vals)

    positive_input_ids = input_ids.clone()
    positive_attention_mask = attention_mask.clone()
    bsz, seq_len = input_ids.size()
    for i in range(bsz):
        seq_length = attention_mask[i].sum().item()
        for j in range(seq_length):
            if norm_scores[i, j] < threshold:
                positive_input_ids[i, j] = tokenizer.mask_token_id

    return positive_input_ids, positive_attention_mask

def train():
    with open('D:/Project/tectClass/configs/config.yaml', 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])

    # 先加载临时数据集，获取 num_labels_l1, num_labels_l2, parent_to_children
    temp_dataset = TextDataset(config['data']['train_dir'], tokenizer, config['model']['max_length'])
    num_labels_l1 = temp_dataset.num_labels_l1
    num_labels_l2 = temp_dataset.num_labels_l2
    parent_to_children = temp_dataset.parent_to_children

    train_loader = create_dataloader(
        config['data']['train_dir'],
        tokenizer,
        config['model']['max_length'],
        config['training']['batch_size'],
        shuffle=True
    )

    # 构建标签图
    label_graph = nx.DiGraph()
    yl1_path = os.path.join(config['data']['train_dir'], 'YL1.txt')
    yl2_path = os.path.join(config['data']['train_dir'], 'YL2.txt')
    if os.path.exists(yl1_path) and os.path.exists(yl2_path):
        with open(yl1_path, 'r', encoding='utf-8') as f:
            level1_labels = [line.strip() for line in f]
        with open(yl2_path, 'r', encoding='utf-8') as f:
            level2_labels = [line.strip() for line in f]
        unique_level1 = sorted(set(level1_labels))
        for parent_label in unique_level1:
            label_graph.add_node(parent_label)
        for (p, c) in zip(level1_labels, level2_labels):
            label_graph.add_node(c)
            label_graph.add_edge(p, c)

    # 构造邻接矩阵
    nodes_list = sorted(label_graph.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes_list)}
    adjacency_matrix = torch.zeros(len(nodes_list), len(nodes_list))
    for (u, v) in label_graph.edges():
        i, j = node_index_map[u], node_index_map[v]
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1

    model = HierarchicalTextModel(
        bert_model_name=config['model']['bert_model_name'],
        num_labels_l1=num_labels_l1,
        num_labels_l2=num_labels_l2,
        label_embedding_dim=config['model']['label_embedding_dim'],
        label_graph=label_graph,
        adjacency_matrix=adjacency_matrix
    ).to(device)

    criterion_l1 = nn.CrossEntropyLoss()
    criterion_l2 = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])

    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    writer = SummaryWriter(log_dir='logs/experiment')
    num_epochs = config['training']['num_epochs']
    temperature = config['training']['temperature']
    contrastive_loss_weight = 0.5

    history = {
        'train_loss': [],
        'train_cls_loss_l1': [],
        'train_cls_loss_l2': [],
        'train_contrastive_loss': []
    }

    model_save_counter = 1
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_l2_loss = 0.0
        epoch_ctr_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_l1 = batch['labels_l1'].to(device)
            labels_l2 = batch['labels_l2'].to(device)

            positive_input_ids, positive_attention_mask = generate_positive_samples(
                model, tokenizer, batch, threshold=0.3
            )

            logits_l1, logits_l2, z_i, z_j = model(
                input_ids, attention_mask,
                positive_input_ids, positive_attention_mask
            )

            # l1 损失
            l1_loss = criterion_l1(logits_l1, labels_l1)

            # l2 损失：对 l2 做父标签限制
            masked_logits_l2 = []
            for i in range(input_ids.size(0)):
                parent = labels_l1[i].item()
                allowed = parent_to_children[parent]
                mask = torch.full((num_labels_l2,), float(-1e9), device=device)
                for idx in allowed:
                    mask[idx] = 0
                masked_logits_l2.append(logits_l2[i] + mask)
            masked_logits_l2 = torch.stack(masked_logits_l2)
            l2_loss = criterion_l2(masked_logits_l2, labels_l2)

            cls_loss = l1_loss + l2_loss

            # 对比损失
            ctr_loss = nt_xent_loss(z_i, z_j, temperature)
            total_loss = cls_loss + contrastive_loss_weight * ctr_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_l2_loss += l2_loss.item()
            epoch_ctr_loss += ctr_loss.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_l1_loss = epoch_l1_loss / len(train_loader)
        avg_l2_loss = epoch_l2_loss / len(train_loader)
        avg_ctr_loss = epoch_ctr_loss / len(train_loader)

        history['train_loss'].append(avg_loss)
        history['train_cls_loss_l1'].append(avg_l1_loss)
        history['train_cls_loss_l2'].append(avg_l2_loss)
        history['train_contrastive_loss'].append(avg_ctr_loss)

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Cls_L1_Loss/train', avg_l1_loss, epoch)
        writer.add_scalar('Cls_L2_Loss/train', avg_l2_loss, epoch)
        writer.add_scalar('Ctr_Loss/train', avg_ctr_loss, epoch)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Total Loss: {avg_loss:.4f} | L1: {avg_l1_loss:.4f} "
            f"| L2: {avg_l2_loss:.4f} | CTR: {avg_ctr_loss:.4f}"
        )

        # 每训练2轮保存一次模型
        if (epoch + 1) % 2 == 0:
            model_save_path = f"D:/Project/tectClass/models/hierarchical_model_multiout_2_20_40000{model_save_counter}.pth"
            torch.save(model.state_dict(), model_save_path)
            model_save_counter += 1

    final_save_path = "D:/Project/tectClass/models/hierarchical_model_multiout_final_4_4_9000_4.pth"
    torch.save(model.state_dict(), final_save_path)
    writer.close()

    plot_metric_curve({
        'Train_Total_Loss': history['train_loss'],
        'Train_Cls_L1_Loss': history['train_cls_loss_l1'],
        'Train_Cls_L2_Loss': history['train_cls_loss_l2'],
        'Train_Contrastive_Loss': history['train_contrastive_loss']
    }, title='Training Loss Curves')


if __name__ == '__main__':
    train()
```

### 4. 评估脚本 (`scripts/evaluate.py`)

```python
import torch
import yaml
import os
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertTokenizer

from utils.data_loader import create_dataloader, TextDataset
from utils.model_utils import HierarchicalTextModel

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
def load_training_mapping(train_dir):
    """
    从训练目录加载 YL1 与 YL2 标签，并构造 global_label2id 和 parent_to_children，
    保证训练和评估时 l2 的全局编码一致。
    """
    with open(os.path.join(train_dir, 'YL1.txt'), 'r', encoding='utf-8') as f:
        level1_labels = [int(line.strip()) for line in f]
    with open(os.path.join(train_dir, 'YL2.txt'), 'r', encoding='utf-8') as f:
        raw_level2 = [line.strip() for line in f]
    global_label2id = {}
    parent_to_children = {}
    for l1, l2_str in zip(level1_labels, raw_level2):
        l2_int = int(l2_str)
        key = (l1, l2_int)
        if key not in global_label2id:
            global_label2id[key] = len(global_label2id)
        if l1 not in parent_to_children:
            parent_to_children[l1] = set()
        parent_to_children[l1].add(global_label2id[key])
    return global_label2id, parent_to_children


def evaluate():
    # 加载配置文件
    with open('D:/Project/tectClass/configs/config.yaml', 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 设置设备和 tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])

    # 加载训练时构造的 mapping 保证 l2 全局编码和 l1->l2 映射在评估时一致
    global_mapping, parent_to_children = load_training_mapping(config['data']['train_dir'])

    # 创建评估 DataLoader，指定 mapping
    eval_loader = create_dataloader(
        config['data']['eval_dir'],
        tokenizer,
        config['model']['max_length'],
        config['training']['batch_size'],
        shuffle=False,
        global_label2id=global_mapping,
        parent_to_children=parent_to_children
    )

    # 构建标签图（与训练时保持一致，用于 Graphormer；可根据实际情况选择是否构造）
    label_graph = nx.DiGraph()
    yl1_path = os.path.join(config['data']['train_dir'], 'YL1.txt')
    yl2_path = os.path.join(config['data']['train_dir'], 'YL2.txt')
    if os.path.exists(yl1_path) and os.path.exists(yl2_path):
        with open(yl1_path, 'r', encoding='utf-8') as f:
            level1_labels = [line.strip() for line in f]
        with open(yl2_path, 'r', encoding='utf-8') as f:
            level2_labels = [line.strip() for line in f]
        for p, c in zip(level1_labels, level2_labels):
            label_graph.add_node(p)
            label_graph.add_node(c)
            label_graph.add_edge(p, c)

    nodes_list = sorted(label_graph.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes_list)}
    adjacency_matrix = torch.zeros(len(nodes_list), len(nodes_list))
    for (u, v) in label_graph.edges():
        i, j = node_index_map[u], node_index_map[v]
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1

    # 从训练数据中获取 l1 类别数量；l2 类别数量由 global_mapping 获得
    num_labels_l1 = len(set([int(x) for x in open(yl1_path, encoding='utf-8').read().splitlines()]))
    num_labels_l2 = len(global_mapping)

    # 初始化模型
    model = HierarchicalTextModel(
        bert_model_name=config['model']['bert_model_name'],
        num_labels_l1=num_labels_l1,
        num_labels_l2=num_labels_l2,
        label_embedding_dim=config['model']['label_embedding_dim'],
        label_graph=label_graph,
        adjacency_matrix=adjacency_matrix
    )

    # 加载训练好的模型权重
    model_path = "D:/Project/tectClass/models/hierarchical_model_multiout_final_4_4_9000_4.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds_l1 = []
    all_preds_l2 = []
    all_labels_l1 = []
    all_labels_l2 = []

    # 评估过程：先预测 l1，然后利用预测 l1 构造 l2 的候选范围进行预测。
    # 这里，我们进一步修正：如果 l1 预测错误，则将该样本的 l2 预测置为错误（全部设置为 -∞）。
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lbl_l1 = batch['labels_l1'].to(device)
            lbl_l2 = batch['labels_l2'].to(device)

            logits_l1, logits_l2, _, _ = model(input_ids, attention_mask)
            # 预测 l1
            preds_l1 = torch.argmax(logits_l1, dim=1)

            # 利用预测的 l1 构造 mask，限定 l2 预测范围
            masked_logits_l2 = []
            for i in range(input_ids.size(0)):
                predicted_parent = preds_l1[i].item()
                allowed_children = parent_to_children.get(predicted_parent, set())
                mask = torch.full((num_labels_l2,), float(-1e9), device=device)
                for idx in allowed_children:
                    mask[idx] = 0
                # 如果 l1 预测错误，则将 mask 全部设置为 -∞，保证 l2 预测错误
                if preds_l1[i].item() != lbl_l1[i].item():
                    mask = torch.full((num_labels_l2,), float(-1e9), device=device)
                masked_logits_l2.append(logits_l2[i] + mask)
            masked_logits_l2 = torch.stack(masked_logits_l2)
            preds_l2 = torch.argmax(masked_logits_l2, dim=1)

            print(preds_l1)
            print(preds_l2)
            print(lbl_l1)
            print(lbl_l2)


            all_preds_l1.extend(preds_l1.cpu().tolist())
            all_preds_l2.extend(preds_l2.cpu().tolist())
            all_labels_l1.extend(lbl_l1.cpu().tolist())
            all_labels_l2.extend(lbl_l2.cpu().tolist())

    # 计算各项指标，其中 Joint 只有在 l1 和 l2 均预测正确时才认为正确
    acc_l1 = accuracy_score(all_labels_l1, all_preds_l1)
    acc_l2 = accuracy_score(all_labels_l2, all_preds_l2)
    joint_correct = sum(1 for pl1, pl2, gl1, gl2 in zip(all_preds_l1, all_preds_l2, all_labels_l1, all_labels_l2)
                        if (pl1 == gl1 and pl2 == gl2))
    joint_acc = joint_correct / len(all_preds_l1)

    joint_preds = [f"{p1}_{p2}" for p1, p2 in zip(all_preds_l1, all_preds_l2)]
    joint_labels = [f"{g1}_{g2}" for g1, g2 in zip(all_labels_l1, all_labels_l2)]
    joint_precision_micro = precision_score(joint_labels, joint_preds, average='weighted')
    joint_precision_macro = precision_score(joint_labels, joint_preds, average='weighted')
    joint_recall_micro = recall_score(joint_labels, joint_preds, average='weighted')
    joint_recall_macro = recall_score(joint_labels, joint_preds, average='weighted')
    joint_f1_micro = f1_score(joint_labels, joint_preds, average='micro')
    joint_f1_macro = f1_score(joint_labels, joint_preds, average='macro')

    print("Level 1 (YL1) Metrics:")
    print(f"  Accuracy: {acc_l1:.4f}")
    print(f"  F1 Score (Weighted): {f1_score(all_labels_l1, all_preds_l1, average='weighted'):.4f}")

    print("Level 2 (YL2) Metrics:")
    print(f"  Accuracy: {acc_l2:.4f}")
    print(f"  F1 Score (Weighted): {f1_score(all_labels_l2, all_preds_l2, average='weighted'):.4f}")

    print("Joint Metrics (Both Levels):")
    print(f"  Joint Accuracy: {joint_acc:.4f}")
    print(f"  Precision: Micro={joint_precision_micro:.4f}, Macro={joint_precision_macro:.4f}")
    print(f"  Recall:    Micro={joint_recall_micro:.4f}, Macro={joint_recall_macro:.4f}")
    print(f"  F1 Score:  Micro={joint_f1_micro:.4f}, Macro={joint_f1_macro:.4f}")


if __name__ == '__main__':
    evaluate()
```

### 5. 可视化工具 (`utils/visualization.py`)

```python
import matplotlib.pyplot as plt

def plot_metric_curve(metric_dict, title='Training Curve'):
    """
    绘制训练过程中Loss、Accuracy等曲线。
    metric_dict示例:
      {
        'train_loss': [...],
        'train_cls_loss_l1': [...],
        'train_cls_loss_l2': [...],
        'train_contrastive_loss': [...]
      }
    """
    plt.figure(figsize=(10, 6))
    for metric_name, vals in metric_dict.items():
        plt.plot(vals, label=metric_name)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

def plot_contrastive_loss(train_contrastive_list, val_contrastive_list=None):
    """
    绘制对比学习损失曲线
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_contrastive_list, label='Train Contrastive Loss')
    if val_contrastive_list is not None:
        plt.plot(val_contrastive_list, label='Val Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.title('Contrastive Loss')
    plt.legend()
    plt.show()
```

### 6. 配置文件 (`configs/config.yaml`)

配置文件可以根据需要调整超参数：

```yaml
model:
  bert_model_name: 'bert-base-uncased'
  max_length: 256
  label_embedding_dim: 128

training:
  batch_size: 8
  learning_rate: 0.00002
  num_epochs: 1
  temperature: 0.5

data:
  train_dir: 'D:/Project/tectClass/data/wos-train-40000'
  eval_dir: 'D:/Project/tectClass/data/wos-evaluate-40000'
```