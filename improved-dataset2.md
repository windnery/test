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
from torch.utils.data import DataLoader

class TextDataset(Dataset):
    """
    处理三级层次文本分类数据集，将文本标签转换为数值ID。
    构建完整的层次结构映射，确保预测时能够遵循层次约束。
    包含分层提示(hierarchical_prompt)来提高模型的层次感知能力。
    """

    def __init__(self, data_dir, tokenizer, max_length,
                 level1_to_id=None, level2_to_id=None, level3_to_id=None,
                 level1_to_level2_map=None, level2_to_level3_map=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 读取文本和三级标签
        with open(os.path.join(data_dir, 'X.txt'), 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f]

        with open(os.path.join(data_dir, 'YL1.txt'), 'r', encoding='utf-8') as f:
            self.level1_labels_text = [line.strip() for line in f]

        with open(os.path.join(data_dir, 'YL2.txt'), 'r', encoding='utf-8') as f:
            self.level2_labels_text = [line.strip() for line in f]

        with open(os.path.join(data_dir, 'YL3.txt'), 'r', encoding='utf-8') as f:
            self.level3_labels_text = [line.strip() for line in f]

        # 验证所有文件的行数是否一致
        assert len(self.texts) == len(self.level1_labels_text) == len(self.level2_labels_text) == len(
            self.level3_labels_text), \
            "文本和标签文件的行数不匹配"

        # 如果提供了映射，就使用提供的映射，否则根据当前数据计算
        if level1_to_id is None:
            # 构建标签到ID的映射
            self.level1_to_id = {label: idx for idx, label in enumerate(sorted(set(self.level1_labels_text)))}
            self.level2_to_id = {label: idx for idx, label in enumerate(sorted(set(self.level2_labels_text)))}
            self.level3_to_id = {label: idx for idx, label in enumerate(sorted(set(self.level3_labels_text)))}

            # 构建层级关系映射
            self.level1_to_level2_map = {}
            self.level2_to_level3_map = {}

            for l1_text, l2_text, l3_text in zip(self.level1_labels_text, self.level2_labels_text,
                                                 self.level3_labels_text):
                l1_id = self.level1_to_id[l1_text]
                l2_id = self.level2_to_id[l2_text]
                l3_id = self.level3_to_id[l3_text]

                if l1_id not in self.level1_to_level2_map:
                    self.level1_to_level2_map[l1_id] = set()
                self.level1_to_level2_map[l1_id].add(l2_id)

                if l2_id not in self.level2_to_level3_map:
                    self.level2_to_level3_map[l2_id] = set()
                self.level2_to_level3_map[l2_id].add(l3_id)
        else:
            self.level1_to_id = level1_to_id
            self.level2_to_id = level2_to_id
            self.level3_to_id = level3_to_id
            self.level1_to_level2_map = level1_to_level2_map
            self.level2_to_level3_map = level2_to_level3_map

        # 转换文本标签为ID
        self.level1_ids = [self.level1_to_id[label] for label in self.level1_labels_text]
        self.level2_ids = [self.level2_to_id[label] for label in self.level2_labels_text]
        self.level3_ids = [self.level3_to_id[label] for label in self.level3_labels_text]

        # 记录每一级的类别数
        self.num_labels_l1 = len(self.level1_to_id)
        self.num_labels_l2 = len(self.level2_to_id)
        self.num_labels_l3 = len(self.level3_to_id)

        # 创建反向映射（ID到文本标签）
        self.id_to_level1 = {v: k for k, v in self.level1_to_id.items()}
        self.id_to_level2 = {v: k for k, v in self.level2_to_id.items()}
        self.id_to_level3 = {v: k for k, v in self.level3_to_id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        l1_text = self.level1_labels_text[idx]
        l2_text = self.level2_labels_text[idx]
        l3_text = self.level3_labels_text[idx]
        l1_id = self.level1_ids[idx]
        l2_id = self.level2_ids[idx]
        l3_id = self.level3_ids[idx]

        # 构造层次提示，将文本与标签信息结合
        # 按照要求的格式: "{text} [SEP] Level1: {yl1} [SEP] Level2: {yl2}"
        # 扩展到三级: "{text} [SEP] Level1: {yl1} [SEP] Level2: {yl2} [SEP] Level3: {yl3}"
        hierarchical_prompt = f"{text} [SEP] Level1: {l1_text} [SEP] Level2: {l2_text} [SEP] Level3: {l3_text}"

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
        item['labels_l1'] = torch.tensor(l1_id, dtype=torch.long)
        item['labels_l2'] = torch.tensor(l2_id, dtype=torch.long)
        item['labels_l3'] = torch.tensor(l3_id, dtype=torch.long)

        # 保存原始文本和标签文本用于调试和分析
        item['text'] = text
        item['l1_text'] = l1_text
        item['l2_text'] = l2_text
        item['l3_text'] = l3_text

        return item



# 增加num_workers和pin_memory
def create_dataloader(data_dir, tokenizer, max_length, batch_size, shuffle=True,
                     level1_to_id=None, level2_to_id=None, level3_to_id=None,
                     level1_to_level2_map=None, level2_to_level3_map=None):
    dataset = TextDataset(
        data_dir,
        tokenizer,
        max_length,
        level1_to_id,
        level2_to_id,
        level3_to_id,
        level1_to_level2_map,
        level2_to_level3_map
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,  # 增加工作线程数，根据CPU核心数调整
        pin_memory=True,  # 加速CPU到GPU的数据传输
        prefetch_factor=2,  # 预加载的批次数
        persistent_workers=True  # 保持工作进程活跃
    ), dataset
```

### 2. 模型定义 (`utils/model_utils.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import math
from collections import deque
import networkx as nx


def compute_shortest_distances(adj_matrix):
    """优化的最短路径计算"""
    device = adj_matrix.device

    # 如果是在CUDA上，尽量避免CPU-GPU传输
    if device.type == 'cuda':
        # 使用Floyd-Warshall算法在GPU上计算
        num_nodes = adj_matrix.size(0)
        # 初始化距离矩阵
        dist = torch.where(adj_matrix > 0,
                           adj_matrix.float(),
                           torch.tensor(float('inf'), device=device))
        # 设置对角线为0
        dist.fill_diagonal_(0)

        # Floyd-Warshall算法
        for k in range(num_nodes):
            # 并行计算
            dist = torch.min(
                dist,
                dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0)
            )

        # 处理无穷大值
        dist[torch.isinf(dist)] = num_nodes + 1
        return dist
    else:
        # 在CPU上使用原始BFS实现但做一些优化
        adj_cpu = adj_matrix.cpu()
        num_nodes = adj_cpu.size(0)
        distances_cpu = torch.full((num_nodes, num_nodes), float('inf'))

        # 获取稀疏表示以加速邻居搜索
        edges = adj_cpu.nonzero(as_tuple=True)
        neighbors = [[] for _ in range(num_nodes)]
        for i, j in zip(edges[0], edges[1]):
            neighbors[i.item()].append(j.item())

        for i in range(num_nodes):
            distances_cpu[i, i] = 0
            visited = [False] * num_nodes
            queue = deque([(i, 0)])
            visited[i] = True

            while queue:
                current, dist = queue.popleft()
                distances_cpu[i, current] = dist
                # 只遍历实际邻居而不是所有节点
                for neighbor in neighbors[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, dist + 1))

        distances_cpu[torch.isinf(distances_cpu)] = num_nodes + 1
        return distances_cpu.to(device)


class Graphormer(nn.Module):
    """
    适用于标签图的Graphormer，支持：
    1) 边缘编码 (edge encoding)
    2) 距离编码 (distance encoding)
    3) 在注意力中加入空间偏置 (spatial bias)
    """

    def __init__(self, hidden_size, nhead=8, edge_types=2, max_distance=10, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.edge_types = edge_types
        self.max_distance = max_distance
        self.num_layers = num_layers

        # 边类型embedding
        self.edge_type_embedding = nn.Embedding(self.edge_types, hidden_size)
        # 距离embedding
        self.distance_embedding = nn.Embedding(self.max_distance + 2, hidden_size)
        # 多头空间偏置
        self.spatial_bias = nn.Parameter(torch.zeros(self.nhead, hidden_size))

        # 增强版TransformerEncoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 规范化层
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, label_embeddings, adjacency_matrix=None):
        # [num_nodes, hidden_size] -> [1, num_nodes, hidden_size]
        label_emb = label_embeddings.unsqueeze(0)
        label_emb = self.norm(label_emb)  # 输入规范化

        if adjacency_matrix is None:
            out = self.encoder(label_emb)
            return out.squeeze(0)

        # 计算最短距离
        distances = compute_shortest_distances(adjacency_matrix)
        distances_clamped = distances.clamp(max=self.max_distance)

        # 创建注意力掩码
        attention_mask = (adjacency_matrix == 0).unsqueeze(0)

        # 边类型编码
        edge_type_tensor = adjacency_matrix.clone().long()
        edge_type_tensor = edge_type_tensor.clamp(0, self.edge_types - 1)
        edge_type_emb = self.edge_type_embedding(edge_type_tensor)

        # 距离编码
        distance_tensor = distances_clamped.long()
        distance_emb = self.distance_embedding(distance_tensor)

        # 组合空间编码
        spatial_enc = edge_type_emb + distance_emb

        # 将空间编码与注意力机制融合
        spatial_biases = []
        for i in range(self.nhead):
            # 每个注意力头可以有不同的空间偏置
            head_bias = torch.einsum('ijh,h->ij', spatial_enc, self.spatial_bias[i])
            spatial_biases.append(head_bias)

        # 堆叠所有头的空间偏置
        stacked_spatial_biases = torch.stack(spatial_biases)

        # 归一化处理
        stacked_spatial_biases = stacked_spatial_biases / math.sqrt(self.hidden_size)

        # 扩展attention_mask为[nhead, num_nodes, num_nodes]
        expanded_attention_mask = attention_mask.expand(self.nhead, -1, -1)

        # 添加空间偏置到attention mask
        combined_mask = expanded_attention_mask.float()
        combined_mask = combined_mask + (-stacked_spatial_biases)

        # 通过Transformer编码
        out = self.encoder(label_emb, mask=combined_mask.bool())
        return out.squeeze(0)


class HierarchicalTextModel(nn.Module):
    """
    三级层次文本分类模型，采用BERT进行文本编码，
    结合Graphormer对标签图进行建模，最终输出三级分类结果。
    包含对比学习模块，提高模型表示能力。
    """

    def __init__(self, bert_model_name, num_labels_l1, num_labels_l2, num_labels_l3,
                 label_embedding_dim, label_graph=None, dropout=0.1, adjacency_matrix=None):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        # 如果提供了标签图，使用图结构；否则只使用节点数量
        if label_graph:
            num_graph_nodes = len(label_graph.nodes)
            self.label_embedding = nn.Embedding(num_embeddings=num_graph_nodes,
                                                embedding_dim=label_embedding_dim)
        else:
            # 如果没有提供图，则为每一级别的标签创建嵌入
            total_labels = num_labels_l1 + num_labels_l2 + num_labels_l3
            self.label_embedding = nn.Embedding(num_embeddings=total_labels,
                                                embedding_dim=label_embedding_dim)

        # 图形编码器 - 使用Graphormer处理标签依赖关系
        self.graph_encoder = Graphormer(
            hidden_size=label_embedding_dim,
            nhead=8,
            dropout=dropout,
            num_layers=3  # 增加层数提高建模能力
        )

        if adjacency_matrix is not None:
            self.register_buffer("adjacency_matrix", adjacency_matrix)
        else:
            self.adjacency_matrix = None

        # 标签特征投影
        self.label_linear = nn.Linear(label_embedding_dim, self.hidden_size)

        # 文本特征处理
        self.feature_dropout = nn.Dropout(dropout)

        # 对比学习投影头 - 增强表示
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # 三个分类器分别预测l1, l2和l3
        self.classifier_l1 = nn.Linear(self.hidden_size, num_labels_l1)
        self.classifier_l2 = nn.Linear(self.hidden_size, num_labels_l2)
        self.classifier_l3 = nn.Linear(self.hidden_size, num_labels_l3)

        # 层次分类器 - 促进层次依赖关系学习
        self.hierarchical_l1_to_l2 = nn.Linear(self.hidden_size, num_labels_l2)
        self.hierarchical_l2_to_l3 = nn.Linear(self.hidden_size, num_labels_l3)

        # 层次注意力 - 聚焦于标签依赖信息
        self.hierarchy_attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 3)  # 三个层次的注意力权重
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                positive_input_ids=None, positive_attention_mask=None, positive_token_type_ids=None):
        # 编码输入文本
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.feature_dropout(cls_emb)

        # 获取标签特征
        label_features = self.get_label_features()
        label_features = self.label_linear(label_features)

        # 组合文本和标签特征
        combined_features = cls_emb + label_features

        # 计算层次注意力权重
        hierarchy_weights = F.softmax(self.hierarchy_attention(combined_features), dim=-1)

        # 三级分类预测
        logits_l1 = self.classifier_l1(combined_features)

        # 利用L1预测增强L2预测
        l1_features = F.softmax(logits_l1, dim=-1)
        hierarchical_l2_logits = self.hierarchical_l1_to_l2(combined_features)
        logits_l2 = self.classifier_l2(combined_features) + hierarchy_weights[:, 0].unsqueeze(
            -1) * hierarchical_l2_logits

        # 利用L2预测增强L3预测
        l2_features = F.softmax(logits_l2, dim=-1)
        hierarchical_l3_logits = self.hierarchical_l2_to_l3(combined_features)
        logits_l3 = self.classifier_l3(combined_features) + hierarchy_weights[:, 1].unsqueeze(
            -1) * hierarchical_l3_logits

        # 对比学习表示
        z_i = self.projection_head(combined_features)
        z_j = None

        # 处理正样本对（如果提供）
        if positive_input_ids is not None and positive_attention_mask is not None:
            pos_outputs = self.encoder(
                input_ids=positive_input_ids,
                attention_mask=positive_attention_mask,
                token_type_ids=positive_token_type_ids,
                output_attentions=True
            )
            pos_cls_emb = pos_outputs.last_hidden_state[:, 0, :]
            pos_cls_emb = self.feature_dropout(pos_cls_emb)
            pos_combined_features = pos_cls_emb + label_features
            z_j = self.projection_head(pos_combined_features)

        return logits_l1, logits_l2, logits_l3, z_i, z_j

    def get_label_features(self):
        """获取通过图结构编码的标签特征"""
        label_embeddings = self.label_embedding.weight
        if self.adjacency_matrix is not None:
            encoded_label_features = self.graph_encoder(label_embeddings, self.adjacency_matrix)
        else:
            encoded_label_features = self.graph_encoder(label_embeddings, None)

        # 聚合标签特征，可以根据需要选择不同聚合方式
        # 这里使用平均池化
        label_features = torch.mean(encoded_label_features, dim=0)
        return label_features


class LabelHierarchyGraph:
    """构建标签层次结构图"""

    def __init__(self, level1_texts, level2_texts, level3_texts,
                 level1_to_id, level2_to_id, level3_to_id):
        self.graph = nx.DiGraph()

        # 添加所有节点
        # 层级1节点
        for l1 in level1_to_id:
            self.graph.add_node(level1_to_id[l1], name=l1, level=1)

        # 层级2节点 - 节点ID需要偏移，避免与层级1冲突
        l1_offset = len(level1_to_id)
        for l2 in level2_to_id:
            self.graph.add_node(level2_to_id[l2] + l1_offset, name=l2, level=2)

        # 层级3节点 - 节点ID需要偏移，避免与层级1和层级2冲突
        l2_offset = l1_offset + len(level2_to_id)
        for l3 in level3_to_id:
            self.graph.add_node(level3_to_id[l3] + l2_offset, name=l3, level=3)

        # 添加边 (L1->L2, L2->L3)
        # 同时记录出现的配对关系
        l1l2_pairs = set()
        l2l3_pairs = set()

        for l1_text, l2_text, l3_text in zip(level1_texts, level2_texts, level3_texts):
            # 获取节点ID
            l1_id = level1_to_id[l1_text]
            l2_id = level2_to_id[l2_text] + l1_offset
            l3_id = level3_to_id[l3_text] + l2_offset

            # 添加L1-L2边
            l1l2_pair = (l1_id, l2_id)
            if l1l2_pair not in l1l2_pairs:
                self.graph.add_edge(l1_id, l2_id, weight=1.0)
                l1l2_pairs.add(l1l2_pair)

            # 添加L2-L3边
            l2l3_pair = (l2_id, l3_id)
            if l2l3_pair not in l2l3_pairs:
                self.graph.add_edge(l2_id, l3_id, weight=1.0)
                l2l3_pairs.add(l2l3_pair)

    def get_adjacency_matrix(self):
        """获取图的邻接矩阵"""
        num_nodes = len(self.graph.nodes)
        adj_matrix = torch.zeros(num_nodes, num_nodes)

        # 设置有向边
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            adj_matrix[u, v] = weight
            # 双向连接增强信息流动
            adj_matrix[v, u] = weight * 0.5  # 反向边权重可以降低

        return adj_matrix

    def get_graph(self):
        """获取networkx图对象"""
        return self.graph
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
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import random
from datetime import datetime

from utils.data_loader import create_dataloader
from utils.model_utils import HierarchicalTextModel, LabelHierarchyGraph
from utils.visualization import plot_metric_curve

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """NT-Xent损失函数，用于对比学习"""
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


def get_token_importance(model, input_ids, attention_mask, token_type_ids=None):
    """计算句子中各token的重要度"""
    with torch.no_grad():
        outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True
        )
        attentions = outputs.attentions
        all_attentions = torch.stack(attentions, dim=0)
        # 计算所有层、所有头的平均注意力
        mean_attention = all_attentions.mean(dim=(0, 2))
        # 每个token接收到的注意力总和作为重要度
        token_importance = mean_attention.sum(dim=-1)
    return token_importance


def generate_positive_samples(model, tokenizer, batch, threshold=0.3):
    """使用重要度采样替换低重要度token为[MASK]，构造强化正样本"""
    device = next(model.parameters()).device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch.get('token_type_ids', None)

    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
        importance_scores = get_token_importance(model, input_ids, attention_mask, token_type_ids)
    else:
        importance_scores = get_token_importance(model, input_ids, attention_mask)

    # 归一化得分
    min_vals = importance_scores.min(dim=1, keepdim=True)[0]
    max_vals = importance_scores.max(dim=1, keepdim=True)[0].clamp(min=1e-9)
    norm_scores = (importance_scores - min_vals) / (max_vals - min_vals)

    positive_input_ids = input_ids.clone()
    positive_attention_mask = attention_mask.clone()
    positive_token_type_ids = token_type_ids.clone() if token_type_ids is not None else None

    bsz, seq_len = input_ids.size()
    for i in range(bsz):
        seq_length = attention_mask[i].sum().item()
        for j in range(seq_length):
            # 避免mask掉特殊token: [CLS], [SEP], Level1:, Level2:, Level3:
            if norm_scores[i, j] < threshold and input_ids[i, j] not in [tokenizer.cls_token_id,
                                                                         tokenizer.sep_token_id]:
                # 检查是否是特殊标记文本的一部分
                if not is_special_text_token(tokenizer, input_ids[i], j):
                    positive_input_ids[i, j] = tokenizer.mask_token_id

    return positive_input_ids, positive_attention_mask, positive_token_type_ids


def is_special_text_token(tokenizer, token_ids, position):
    """检查token是否是特殊文本(如'Level1:')的一部分"""
    special_texts = ["Level1:", "Level2:", "Level3:"]
    for special_text in special_texts:
        # 获取特殊文本的token_ids
        special_token_ids = tokenizer.encode(special_text, add_special_tokens=False)
        # 检查当前位置是否是特殊文本的开始
        if position + len(special_token_ids) <= len(token_ids):
            is_match = True
            for i, token_id in enumerate(special_token_ids):
                if token_ids[position + i].item() != token_id:
                    is_match = False
                    break
            if is_match:
                return True
    return False


def train():
    # 加载配置
    with open('D:/Project/tectClass/configs/config.yaml', 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 设置随机种子
    set_seed(config['training']['seed'])

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建目录
    os.makedirs(config['model']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])

    # 加载数据集
    print("Loading training dataset...")
    train_loader, train_dataset = create_dataloader(
        config['data']['train_dir'],
        tokenizer,
        config['model']['max_length'],
        config['training']['batch_size'],
        shuffle=True
    )

    # 获取标签信息
    num_labels_l1 = train_dataset.num_labels_l1
    num_labels_l2 = train_dataset.num_labels_l2
    num_labels_l3 = train_dataset.num_labels_l3
    level1_to_level2_map = train_dataset.level1_to_level2_map
    level2_to_level3_map = train_dataset.level2_to_level3_map

    print(f"Number of classes: L1={num_labels_l1} ({', '.join(list(train_dataset.level1_to_id.keys())[:3])}...)")
    print(f"Number of classes: L2={num_labels_l2} ({', '.join(list(train_dataset.level2_to_id.keys())[:3])}...)")
    print(f"Number of classes: L3={num_labels_l3} ({', '.join(list(train_dataset.level3_to_id.keys())[:3])}...)")

    # 构建标签层次图
    print("Building label hierarchy graph...")
    label_hierarchy = LabelHierarchyGraph(
        train_dataset.level1_labels_text,
        train_dataset.level2_labels_text,
        train_dataset.level3_labels_text,
        train_dataset.level1_to_id,
        train_dataset.level2_to_id,
        train_dataset.level3_to_id
    )
    adjacency_matrix = label_hierarchy.get_adjacency_matrix().to(device)
    label_graph = label_hierarchy.get_graph()

    print(f"Label graph built with {len(label_graph.nodes)} nodes and {len(label_graph.edges)} edges")

    # 初始化模型
    print("Initializing model...")
    model = HierarchicalTextModel(
        bert_model_name=config['model']['bert_model_name'],
        num_labels_l1=num_labels_l1,
        num_labels_l2=num_labels_l2,
        num_labels_l3=num_labels_l3,
        label_embedding_dim=config['model']['label_embedding_dim'],
        label_graph=label_graph,
        dropout=config['model']['dropout'],
        adjacency_matrix=adjacency_matrix
    ).to(device)

    # 设置优化器
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config['training']['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config['training']['learning_rate'],
        eps=float(config['training']['adam_epsilon'])  # 确保转换为浮点数
    )

    # 设置学习率调度器
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config['training']['warmup_ratio'] * total_steps),
        num_training_steps=total_steps
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # TensorBoard写入器
    writer = SummaryWriter(log_dir=config['logging']['log_dir'])

    # 训练参数
    num_epochs = config['training']['num_epochs']
    temperature = config['training']['temperature']
    contrastive_loss_weight = config['training']['contrastive_loss_weight']

    # 历史记录
    history = {
        'train_loss': [],
        'train_cls_loss_l1': [],
        'train_cls_loss_l2': [],
        'train_cls_loss_l3': [],
        'train_contrastive_loss': []
    }

    # 记录训练开始时间
    start_time = datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 训练循环
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_l2_loss = 0.0
        epoch_l3_loss = 0.0
        epoch_ctr_loss = 0.0

        # 训练进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            labels_l1 = batch['labels_l1'].to(device)
            labels_l2 = batch['labels_l2'].to(device)
            labels_l3 = batch['labels_l3'].to(device)

            # 生成正样本对进行对比学习
            positive_input_ids, positive_attention_mask, positive_token_type_ids = generate_positive_samples(
                model, tokenizer, batch, threshold=config['training']['mask_threshold']
            )

            # 前向传播
            logits_l1, logits_l2, logits_l3, z_i, z_j = model(
                input_ids, attention_mask, token_type_ids,
                positive_input_ids, positive_attention_mask, positive_token_type_ids
            )

            # L1损失
            l1_loss = criterion(logits_l1, labels_l1)

            # L2损失：对L2做父标签约束
            masked_logits_l2 = []
            for i in range(input_ids.size(0)):
                parent = labels_l1[i].item()
                allowed = level1_to_level2_map.get(parent, set())

                # 创建掩码，将不合法的子类标签设为极小值
                mask = torch.full((num_labels_l2,), float(-1e9), device=device)
                for idx in allowed:
                    mask[idx] = 0

                masked_logits_l2.append(logits_l2[i] + mask)

            masked_logits_l2 = torch.stack(masked_logits_l2)
            l2_loss = criterion(masked_logits_l2, labels_l2)

            # L3损失：对L3做父标签约束
            masked_logits_l3 = []
            for i in range(input_ids.size(0)):
                parent = labels_l2[i].item()
                allowed = level2_to_level3_map.get(parent, set())

                # 创建掩码，将不合法的子类标签设为极小值
                mask = torch.full((num_labels_l3,), float(-1e9), device=device)
                for idx in allowed:
                    mask[idx] = 0

                masked_logits_l3.append(logits_l3[i] + mask)

            masked_logits_l3 = torch.stack(masked_logits_l3)
            l3_loss = criterion(masked_logits_l3, labels_l3)

            # 总分类损失 - 三级损失加权
            cls_loss = l1_loss + config['training']['l2_loss_weight'] * l2_loss + config['training'][
                'l3_loss_weight'] * l3_loss

            # 对比损失
            ctr_loss = nt_xent_loss(z_i, z_j, temperature)

            # 总损失
            total_loss = cls_loss + contrastive_loss_weight * ctr_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['max_grad_norm'])

            # 更新参数
            optimizer.step()
            scheduler.step()

            # 更新损失统计
            epoch_loss += total_loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_l2_loss += l2_loss.item()
            epoch_l3_loss += l3_loss.item()
            epoch_ctr_loss += ctr_loss.item()

            # 记录每n步的损失
            if step % config['logging']['log_steps'] == 0:
                writer.add_scalar('Loss/step', total_loss.item(), epoch * len(train_loader) + step)
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch * len(train_loader) + step)

            # 更新进度条显示
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'l1': f"{l1_loss.item():.4f}",
                'l2': f"{l2_loss.item():.4f}",
                'l3': f"{l3_loss.item():.4f}",
                'ctr': f"{ctr_loss.item():.4f}"
            })

        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        avg_l1_loss = epoch_l1_loss / len(train_loader)
        avg_l2_loss = epoch_l2_loss / len(train_loader)
        avg_l3_loss = epoch_l3_loss / len(train_loader)
        avg_ctr_loss = epoch_ctr_loss / len(train_loader)

        # 记录历史
        history['train_loss'].append(avg_loss)
        history['train_cls_loss_l1'].append(avg_l1_loss)
        history['train_cls_loss_l2'].append(avg_l2_loss)
        history['train_cls_loss_l3'].append(avg_l3_loss)
        history['train_contrastive_loss'].append(avg_ctr_loss)

        # 写入TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/l1', avg_l1_loss, epoch)
        writer.add_scalar('Loss/l2', avg_l2_loss, epoch)
        writer.add_scalar('Loss/l3', avg_l3_loss, epoch)
        writer.add_scalar('Loss/contrastive', avg_ctr_loss, epoch)

        # 显示训练信息
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f} | L1: {avg_l1_loss:.4f} | "
            f"L2: {avg_l2_loss:.4f} | L3: {avg_l3_loss:.4f} | "
            f"CTR: {avg_ctr_loss:.4f}"
        )

        # 定期保存模型
        if (epoch + 1) % config['model']['save_every'] == 0:
            model_save_path = os.path.join(
                config['model']['save_dir'],
                f"hierarchical_model_epoch_{epoch + 1}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'level1_to_id': train_dataset.level1_to_id,
                'level2_to_id': train_dataset.level2_to_id,
                'level3_to_id': train_dataset.level3_to_id,
                'level1_to_level2_map': level1_to_level2_map,
                'level2_to_level3_map': level2_to_level3_map,
            }, model_save_path)
            print(f"Model checkpoint saved to {model_save_path}")

    # 记录训练结束时间
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"Training finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {training_duration}")

    # 保存最终模型
    final_save_path = os.path.join(config['model']['save_dir'], "hierarchical_model_final.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
        'level1_to_id': train_dataset.level1_to_id,
        'level2_to_id': train_dataset.level2_to_id,
        'level3_to_id': train_dataset.level3_to_id,
        'level1_to_level2_map': level1_to_level2_map,
        'level2_to_level3_map': level2_to_level3_map,
        'training_duration': str(training_duration),
    }, final_save_path)
    print(f"Final model saved to {final_save_path}")

    # 关闭TensorBoard写入器
    writer.close()

    # 绘制损失曲线
    plot_metric_curve({
        'Total Loss': history['train_loss'],
        'L1 Loss': history['train_cls_loss_l1'],
        'L2 Loss': history['train_cls_loss_l2'],
        'L3 Loss': history['train_cls_loss_l3'],
        'Contrastive Loss': history['train_contrastive_loss']
    }, title='Training Loss Curves',
        save_path=os.path.join(config['logging']['log_dir'], 'training_loss_curves.png'))


if __name__ == '__main__':
    train()
```

### 4. 评估脚本 (`scripts/evaluate.py`)

```python
import torch
import yaml
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

from utils.data_loader import create_dataloader
from utils.model_utils import HierarchicalTextModel, LabelHierarchyGraph
from utils.visualization import plot_confusion_matrix, plot_prediction_distribution


def evaluate():
    # 加载配置
    with open('configs/config.yaml', 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建评估结果目录
    eval_result_dir = os.path.join(config['logging']['log_dir'], 'eval_results')
    os.makedirs(eval_result_dir, exist_ok=True)

    # 加载保存的模型检查点
    checkpoint_path = os.path.join(config['model']['save_dir'], "hierarchical_model_final.pth")
    if not os.path.exists(checkpoint_path):
        # 如果最终模型不存在，尝试加载最后一个保存的检查点
        epoch_checkpoints = [f for f in os.listdir(config['model']['save_dir'])
                             if f.startswith('hierarchical_model_epoch_')]
        if epoch_checkpoints:
            # 按照epoch编号排序
            epoch_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint_path = os.path.join(config['model']['save_dir'], epoch_checkpoints[-1])
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError("No model checkpoint found!")

    # 加载模型检查点
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 获取标签映射
    level1_to_id = checkpoint['level1_to_id']
    level2_to_id = checkpoint['level2_to_id']
    level3_to_id = checkpoint['level3_to_id']
    level1_to_level2_map = checkpoint['level1_to_level2_map']
    level2_to_level3_map = checkpoint['level2_to_level3_map']

    # 反向映射 (ID到标签)
    id_to_level1 = {v: k for k, v in level1_to_id.items()}
    id_to_level2 = {v: k for k, v in level2_to_id.items()}
    id_to_level3 = {v: k for k, v in level3_to_id.items()}

    num_labels_l1 = len(level1_to_id)
    num_labels_l2 = len(level2_to_id)
    num_labels_l3 = len(level3_to_id)

    print(f"Number of classes: L1={num_labels_l1}, L2={num_labels_l2}, L3={num_labels_l3}")

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])

    # 加载评估数据集
    print("Loading evaluation dataset...")
    eval_loader, eval_dataset = create_dataloader(
        config['data']['eval_dir'],
        tokenizer,
        config['model']['max_length'],
        config['evaluation']['batch_size'],
        shuffle=False,
        level1_to_id=level1_to_id,
        level2_to_id=level2_to_id,
        level3_to_id=level3_to_id,
        level1_to_level2_map=level1_to_level2_map,
        level2_to_level3_map=level2_to_level3_map
    )

    # 构建标签层次图
    print("Building label hierarchy graph...")
    # 收集所有标签文本
    unique_l1_texts = sorted(set(eval_dataset.level1_labels_text))
    unique_l2_texts = sorted(set(eval_dataset.level2_labels_text))
    unique_l3_texts = sorted(set(eval_dataset.level3_labels_text))

    label_hierarchy = LabelHierarchyGraph(
        eval_dataset.level1_labels_text,
        eval_dataset.level2_labels_text,
        eval_dataset.level3_labels_text,
        level1_to_id,
        level2_to_id,
        level3_to_id
    )
    adjacency_matrix = label_hierarchy.get_adjacency_matrix().to(device)
    label_graph = label_hierarchy.get_graph()

    # 初始化模型
    print("Initializing model...")
    model = HierarchicalTextModel(
        bert_model_name=config['model']['bert_model_name'],
        num_labels_l1=num_labels_l1,
        num_labels_l2=num_labels_l2,
        num_labels_l3=num_labels_l3,
        label_embedding_dim=config['model']['label_embedding_dim'],
        label_graph=label_graph,
        dropout=config['model']['dropout'],
        adjacency_matrix=adjacency_matrix
    ).to(device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 初始化指标收集
    all_preds_l1 = []
    all_preds_l2 = []
    all_preds_l3 = []
    all_labels_l1 = []
    all_labels_l2 = []
    all_labels_l3 = []

    # 收集原始文本和预测结果，用于错误分析
    all_texts = []
    all_pred_l1_texts = []
    all_pred_l2_texts = []
    all_pred_l3_texts = []
    all_true_l1_texts = []
    all_true_l2_texts = []
    all_true_l3_texts = []

    # 开始评估
    print("Starting evaluation...")
    evaluation_start_time = datetime.now()

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            labels_l1 = batch['labels_l1'].to(device)
            labels_l2 = batch['labels_l2'].to(device)
            labels_l3 = batch['labels_l3'].to(device)

            # 保存原始文本和标签文本
            all_texts.extend(batch['text'])
            all_true_l1_texts.extend(batch['l1_text'])
            all_true_l2_texts.extend(batch['l2_text'])
            all_true_l3_texts.extend(batch['l3_text'])

            # 前向传播
            logits_l1, logits_l2, logits_l3, _, _ = model(
                input_ids, attention_mask, token_type_ids
            )

            # 预测L1
            preds_l1 = torch.argmax(logits_l1, dim=1)

            # 基于预测的L1约束L2预测
            masked_logits_l2 = []
            for i in range(input_ids.size(0)):
                predicted_parent = preds_l1[i].item()
                allowed_children = level1_to_level2_map.get(predicted_parent, set())

                mask = torch.full((num_labels_l2,), float(-1e9), device=device)
                for idx in allowed_children:
                    mask[idx] = 0
                masked_logits_l2.append(logits_l2[i] + mask)

            masked_logits_l2 = torch.stack(masked_logits_l2)
            preds_l2 = torch.argmax(masked_logits_l2, dim=1)

            # 基于预测的L2约束L3预测
            masked_logits_l3 = []
            for i in range(input_ids.size(0)):
                predicted_parent = preds_l2[i].item()
                allowed_children = level2_to_level3_map.get(predicted_parent, set())

                mask = torch.full((num_labels_l3,), float(-1e9), device=device)
                for idx in allowed_children:
                    mask[idx] = 0

                # 如果L2预测错误（与真实值不符），我们可以选择是否强制L3预测无效
                # 这取决于我们的应用需求，这里我们仍然允许预测
                # if preds_l2[i].item() != labels_l2[i].item():
                #     mask = torch.full((num_labels_l3,), float(-1e9), device=device)

                masked_logits_l3.append(logits_l3[i] + mask)

            masked_logits_l3 = torch.stack(masked_logits_l3)
            preds_l3 = torch.argmax(masked_logits_l3, dim=1)

            # 将预测ID转换为文本标签
            for i in range(len(preds_l1)):
                all_pred_l1_texts.append(id_to_level1[preds_l1[i].item()])
                all_pred_l2_texts.append(id_to_level2[preds_l2[i].item()])
                all_pred_l3_texts.append(id_to_level3[preds_l3[i].item()])

            # 收集预测和真实标签
            all_preds_l1.extend(preds_l1.cpu().tolist())
            all_preds_l2.extend(preds_l2.cpu().tolist())
            all_preds_l3.extend(preds_l3.cpu().tolist())
            all_labels_l1.extend(labels_l1.cpu().tolist())
            all_labels_l2.extend(labels_l2.cpu().tolist())
            all_labels_l3.extend(labels_l3.cpu().tolist())

    # 计算评估时间
    evaluation_end_time = datetime.now()
    evaluation_duration = evaluation_end_time - evaluation_start_time
    print(f"Evaluation completed in {evaluation_duration}")

    # 计算各级别准确率
    accuracy_l1 = accuracy_score(all_labels_l1, all_preds_l1)
    accuracy_l2 = accuracy_score(all_labels_l2, all_preds_l2)
    accuracy_l3 = accuracy_score(all_labels_l3, all_preds_l3)

    # 计算联合准确率 (所有级别同时正确的比例)
    joint_correct = sum(1 for p1, p2, p3, t1, t2, t3 in
                        zip(all_preds_l1, all_preds_l2, all_preds_l3,
                            all_labels_l1, all_labels_l2, all_labels_l3)
                        if p1 == t1 and p2 == t2 and p3 == t3)
    joint_accuracy = joint_correct / len(all_preds_l1)

    # 计算精度、召回率和F1分数
    precision_l1 = precision_score(all_labels_l1, all_preds_l1, average='weighted')
    precision_l2 = precision_score(all_labels_l2, all_preds_l2, average='weighted')
    precision_l3 = precision_score(all_labels_l3, all_preds_l3, average='weighted')

    recall_l1 = recall_score(all_labels_l1, all_preds_l1, average='weighted')
    recall_l2 = recall_score(all_labels_l2, all_preds_l2, average='weighted')
    recall_l3 = recall_score(all_labels_l3, all_preds_l3, average='weighted')

    f1_l1 = f1_score(all_labels_l1, all_preds_l1, average='weighted')
    f1_l2 = f1_score(all_labels_l2, all_preds_l2, average='weighted')
    f1_l3 = f1_score(all_labels_l3, all_preds_l3, average='weighted')

    # 联合指标计算（将三级标签连接成一个标签进行评估）
    joint_pred_labels = [f"{p1}_{p2}_{p3}" for p1, p2, p3 in zip(all_preds_l1, all_preds_l2, all_preds_l3)]
    joint_true_labels = [f"{t1}_{t2}_{t3}" for t1, t2, t3 in zip(all_labels_l1, all_labels_l2, all_labels_l3)]

    joint_precision = precision_score(joint_true_labels, joint_pred_labels, average='weighted', zero_division=0)
    joint_recall = recall_score(joint_true_labels, joint_pred_labels, average='weighted', zero_division=0)
    joint_f1 = f1_score(joint_true_labels, joint_pred_labels, average='weighted', zero_division=0)

    # 输出评估结果
    print("\n===== Evaluation Results =====")
    print(f"Level 1 (YL1) Metrics:")
    print(f"  Accuracy: {accuracy_l1:.4f}")
    print(f"  Precision: {precision_l1:.4f}")
    print(f"  Recall: {recall_l1:.4f}")
    print(f"  F1 Score: {f1_l1:.4f}")

    print(f"\nLevel 2 (YL2) Metrics:")
    print(f"  Accuracy: {accuracy_l2:.4f}")
    print(f"  Precision: {precision_l2:.4f}")
    print(f"  Recall: {recall_l2:.4f}")
    print(f"  F1 Score: {f1_l2:.4f}")

    print(f"\nLevel 3 (YL3) Metrics:")
    print(f"  Accuracy: {accuracy_l3:.4f}")
    print(f"  Precision: {precision_l3:.4f}")
    print(f"  Recall: {recall_l3:.4f}")
    print(f"  F1 Score: {f1_l3:.4f}")

    print(f"\nJoint Metrics (All Levels):")
    print(f"  Joint Accuracy: {joint_accuracy:.4f}")
    print(f"  Joint Precision: {joint_precision:.4f}")
    print(f"  Joint Recall: {joint_recall:.4f}")
    print(f"  Joint F1 Score: {joint_f1:.4f}")

    # 生成详细报告
    print("\n===== Detailed Classification Report =====")
    print("\nLevel 1 (YL1) Report:")
    l1_report = classification_report(
        all_labels_l1, all_preds_l1,
        target_names=[id_to_level1[i] for i in range(num_labels_l1)],
        digits=4
    )
    print(l1_report)

    print("\nLevel 2 (YL2) Report:")
    # 由于L2类别可能很多，只显示前10个
    l2_target_names = [id_to_level2[i] for i in range(min(10, num_labels_l2))]
    if len(l2_target_names) < num_labels_l2:
        l2_target_names.append("...")
    l2_report = classification_report(
        all_labels_l1, all_preds_l1,
        target_names=l2_target_names,
        digits=4
    )
    print(l2_report)

    print("\nLevel 3 (YL3) Report:")
    # 由于L3类别可能很多，只显示前10个
    l3_target_names = [id_to_level3[i] for i in range(min(10, num_labels_l3))]
    if len(l3_target_names) < num_labels_l3:
        l3_target_names.append("...")
    l3_report = classification_report(
        all_labels_l3, all_preds_l3,
        target_names=l3_target_names,
        digits=4
    )
    print(l3_report)

    # 保存评估结果到文件
    result_path = os.path.join(eval_result_dir, "evaluation_results.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("===== Evaluation Results =====\n")
        f.write(f"Level 1 (YL1) Metrics:\n")
        f.write(f"  Accuracy: {accuracy_l1:.4f}\n")
        f.write(f"  Precision: {precision_l1:.4f}\n")
        f.write(f"  Recall: {recall_l1:.4f}\n")
        f.write(f"  F1 Score: {f1_l1:.4f}\n\n")

        f.write(f"Level 2 (YL2) Metrics:\n")
        f.write(f"  Accuracy: {accuracy_l2:.4f}\n")
        f.write(f"  Precision: {precision_l2:.4f}\n")
        f.write(f"  Recall: {recall_l2:.4f}\n")
        f.write(f"  F1 Score: {f1_l2:.4f}\n\n")

        f.write(f"Level 3 (YL3) Metrics:\n")
        f.write(f"  Accuracy: {accuracy_l3:.4f}\n")
        f.write(f"  Precision: {precision_l3:.4f}\n")
        f.write(f"  Recall: {recall_l3:.4f}\n")
        f.write(f"  F1 Score: {f1_l3:.4f}\n\n")

        f.write(f"Joint Metrics (All Levels):\n")
        f.write(f"  Joint Accuracy: {joint_accuracy:.4f}\n")
        f.write(f"  Joint Precision: {joint_precision:.4f}\n")
        f.write(f"  Joint Recall: {joint_recall:.4f}\n")
        f.write(f"  Joint F1 Score: {joint_f1:.4f}\n\n")

        f.write("===== Detailed Classification Report =====\n")
        f.write("\nLevel 1 (YL1) Report:\n")
        f.write(l1_report)

        f.write("\nLevel 2 (YL2) Report:\n")
        f.write(l2_report)

        f.write("\nLevel 3 (YL3) Report:\n")
        f.write(l3_report)

    print(f"\nEvaluation results saved to {result_path}")

    # 创建混淆矩阵可视化
    if config['evaluation']['plot_confusion_matrix']:
        print("Generating confusion matrix visualizations...")
        # L1混淆矩阵（所有类别）
        cm_l1_path = os.path.join(eval_result_dir, 'confusion_matrix_l1.png')
        plot_confusion_matrix(
            all_labels_l1, all_preds_l1,
            class_names=[id_to_level1[i] for i in range(num_labels_l1)],
            title='Level 1 Confusion Matrix',
            save_path=cm_l1_path
        )
        print(f"L1 confusion matrix saved to {cm_l1_path}")

        # L2和L3类别太多，可能不适合直接可视化，可以选择显示最常见的几个类别
        if num_labels_l2 <= 20:  # 如果L2类别数量适中，则绘制完整混淆矩阵
            cm_l2_path = os.path.join(eval_result_dir, 'confusion_matrix_l2.png')
            plot_confusion_matrix(
                all_labels_l2, all_preds_l2,
                class_names=[id_to_level2[i] for i in range(num_labels_l2)],
                title='Level 2 Confusion Matrix',
                save_path=cm_l2_path
            )
            print(f"L2 confusion matrix saved to {cm_l2_path}")
        else:
            # 对于类别过多的情况，只显示最常见的N个类别
            top_n = 15
            l2_counts = pd.Series(all_labels_l2).value_counts().nlargest(top_n).index.tolist()
            l2_mask = np.isin(all_labels_l2, l2_counts) & np.isin(all_preds_l2, l2_counts)

            if np.sum(l2_mask) > 0:
                cm_l2_top_path = os.path.join(eval_result_dir, f'confusion_matrix_l2_top{top_n}.png')
                plot_confusion_matrix(
                    np.array(all_labels_l2)[l2_mask], np.array(all_preds_l2)[l2_mask],
                    class_names=[id_to_level2[i] for i in l2_counts],
                    title=f'Level 2 Confusion Matrix (Top {top_n} Classes)',
                    save_path=cm_l2_top_path
                )
                print(f"L2 top {top_n} classes confusion matrix saved to {cm_l2_top_path}")

    # 绘制预测分布图
    if config['evaluation']['plot_distributions']:
        print("Generating prediction distribution visualizations...")
        # L1预测分布
        dist_l1_path = os.path.join(eval_result_dir, 'prediction_dist_l1.png')
        plot_prediction_distribution(
            all_labels_l1, all_preds_l1,
            class_names=[id_to_level1[i] for i in range(num_labels_l1)],
            title='Level 1 Prediction Distribution',
            save_path=dist_l1_path
        )
        print(f"L1 prediction distribution saved to {dist_l1_path}")

    # 错误分析
    if config['evaluation']['analyze_errors']:
        print("\n===== Error Analysis =====")
        # 找出错误预测的实例
        error_indices = [i for i, (p1, p2, p3, t1, t2, t3) in enumerate(zip(
            all_preds_l1, all_preds_l2, all_preds_l3,
            all_labels_l1, all_labels_l2, all_labels_l3
        )) if p1 != t1 or p2 != t2 or p3 != t3]

        # 选择前N个错误实例进行分析
        n_errors = min(10, len(error_indices))
        selected_errors = error_indices[:n_errors]

        # 创建错误分析报告
        error_report_path = os.path.join(eval_result_dir, "error_analysis.txt")
        with open(error_report_path, 'w', encoding='utf-8') as f:
            f.write("===== Error Analysis =====\n\n")

            for i, idx in enumerate(selected_errors):
                f.write(f"Error #{i + 1}:\n")
                f.write(f"Text: {all_texts[idx][:100]}...\n")
                f.write(f"Level 1: Predicted={all_pred_l1_texts[idx]}, True={all_true_l1_texts[idx]}\n")
                f.write(f"Level 2: Predicted={all_pred_l2_texts[idx]}, True={all_true_l2_texts[idx]}\n")
                f.write(f"Level 3: Predicted={all_pred_l3_texts[idx]}, True={all_true_l3_texts[idx]}\n\n")

            # 统计错误模式
            l1_errors = sum(1 for p1, t1 in zip(all_preds_l1, all_labels_l1) if p1 != t1)
            l2_errors = sum(1 for p2, t2 in zip(all_preds_l2, all_labels_l2) if p2 != t2)
            l3_errors = sum(1 for p3, t3 in zip(all_preds_l3, all_labels_l3) if p3 != t3)

            l1_only_errors = sum(1 for p1, p2, p3, t1, t2, t3 in zip(
                all_preds_l1, all_preds_l2, all_preds_l3,
                all_labels_l1, all_labels_l2, all_labels_l3
            ) if p1 != t1 and p2 == t2 and p3 == t3)

            l2_only_errors = sum(1 for p1, p2, p3, t1, t2, t3 in zip(
                all_preds_l1, all_preds_l2, all_preds_l3,
                all_labels_l1, all_labels_l2, all_labels_l3
            ) if p1 == t1 and p2 != t2 and p3 == t3)

            l3_only_errors = sum(1 for p1, p2, p3, t1, t2, t3 in zip(
                all_preds_l1, all_preds_l2, all_preds_l3,
                all_labels_l1, all_labels_l2, all_labels_l3
            ) if p1 == t1 and p2 == t2 and p3 != t3)

            # 记录错误统计
            f.write("===== Error Statistics =====\n")
            f.write(f"Total Samples: {len(all_labels_l1)}\n")
            f.write(f"Total Errors: {len(error_indices)} ({len(error_indices) / len(all_labels_l1):.2%})\n\n")
            f.write(f"Level 1 Errors: {l1_errors} ({l1_errors / len(all_labels_l1):.2%})\n")
            f.write(f"Level 2 Errors: {l2_errors} ({l2_errors / len(all_labels_l2):.2%})\n")
            f.write(f"Level 3 Errors: {l3_errors} ({l3_errors / len(all_labels_l3):.2%})\n\n")
            f.write(f"Only Level 1 Wrong: {l1_only_errors} ({l1_only_errors / len(all_labels_l1):.2%})\n")
            f.write(f"Only Level 2 Wrong: {l2_only_errors} ({l2_only_errors / len(all_labels_l2):.2%})\n")
            f.write(f"Only Level 3 Wrong: {l3_only_errors} ({l3_only_errors / len(all_labels_l3):.2%})\n")

        print(f"Error analysis report saved to {error_report_path}")

        # 显示一些错误示例
        print("\nSample Error Examples:")
        for i in range(min(5, n_errors)):
            idx = selected_errors[i]
            print(f"\nError #{i + 1}:")
            print(f"Text: {all_texts[idx][:100]}...")
            print(f"Level 1: Predicted={all_pred_l1_texts[idx]}, True={all_true_l1_texts[idx]}")
            print(f"Level 2: Predicted={all_pred_l2_texts[idx]}, True={all_true_l2_texts[idx]}")
            print(f"Level 3: Predicted={all_pred_l3_texts[idx]}, True={all_true_l3_texts[idx]}")

    print("\nEvaluation completed!")
    return {
        'accuracy_l1': accuracy_l1,
        'accuracy_l2': accuracy_l2,
        'accuracy_l3': accuracy_l3,
        'joint_accuracy': joint_accuracy,
        'f1_l1': f1_l1,
        'f1_l2': f1_l2,
        'f1_l3': f1_l3,
        'joint_f1': joint_f1
    }


if __name__ == '__main__':
    evaluate()
```

### 5. 可视化工具 (`utils/visualization.py`)

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import os
from matplotlib.ticker import MaxNLocator
import matplotlib

matplotlib.use('Agg')  # 用于无头服务器


def plot_metric_curve(metric_dict, title='Training Curve', save_path=None):
    """
    绘制训练过程中各种指标曲线。

    Args:
        metric_dict: 包含多个指标值列表的字典
        title: 图表标题
        save_path: 如果提供，保存图表到指定路径
    """
    plt.figure(figsize=(12, 7))

    # 设置风格
    plt.style.use('ggplot')

    # 绘制每个指标
    for metric_name, values in metric_dict.items():
        plt.plot(values, label=metric_name, linewidth=2, marker='o', markersize=4)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置标题和标签
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Value", fontsize=12)

    # 添加图例
    plt.legend(frameon=True, fontsize=10)

    # 美化刻度
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 设置边界填充
    plt.tight_layout()

    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至 {save_path}")

    # 显示图表
    plt.close()


def plot_confusion_matrix(true_labels, pred_labels, class_names=None, title='Confusion Matrix',
                          save_path=None, figsize=(12, 10), normalize=True):
    """
    绘制混淆矩阵。

    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        class_names: 类别名称列表
        title: 图表标题
        save_path: 如果提供，保存图表到指定路径
        figsize: 图表大小
        normalize: 是否归一化
    """
    # 确保类名可用
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(true_labels)))]

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)

    # 归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # 将NaN替换为0

    # 创建热图
    plt.figure(figsize=figsize)

    # 根据类别数量调整字体大小
    if len(class_names) > 20:
        fontsize = 8
    elif len(class_names) > 10:
        fontsize = 10
    else:
        fontsize = 12

    ax = sns.heatmap(cm, annot=False, fmt='.2f' if normalize else 'd',
                     cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    # 设置标签
    plt.ylabel('True Label', fontsize=fontsize + 2)
    plt.xlabel('Predicted Label', fontsize=fontsize + 2)
    plt.title(title, fontsize=fontsize + 4)

    # 设置刻度标签旋转
    plt.xticks(rotation=45, ha='right', fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_prediction_distribution(true_labels, pred_labels, class_names=None, title='Prediction Distribution',
                                 save_path=None, figsize=(14, 8)):
    """
    绘制预测分布图，显示每个类别的真实数量和预测数量。

    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        class_names: 类别名称列表
        title: 图表标题
        save_path: 如果提供，保存图表到指定路径
        figsize: 图表大小
    """
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(np.concatenate([true_labels, pred_labels]))))]

    # 计算每个类别的真实数量和预测数量
    true_counts = pd.Series(true_labels).value_counts().sort_index()
    pred_counts = pd.Series(pred_labels).value_counts().sort_index()

    # 创建数据框
    df = pd.DataFrame({
        'True': pd.Series(0, index=range(len(class_names))),
        'Predicted': pd.Series(0, index=range(len(class_names)))
    })

    for idx in true_counts.index:
        if idx < len(class_names):
            df.loc[idx, 'True'] = true_counts[idx]

    for idx in pred_counts.index:
        if idx < len(class_names):
            df.loc[idx, 'Predicted'] = pred_counts[idx]

    # 绘制分布图
    plt.figure(figsize=figsize)

    # 设置条形图宽度
    bar_width = 0.35
    index = np.arange(len(class_names))

    # 绘制条形图
    plt.bar(index, df['True'], bar_width, label='True', color='steelblue', alpha=0.8)
    plt.bar(index + bar_width, df['Predicted'], bar_width, label='Predicted', color='lightcoral', alpha=0.8)

    # 设置图表属性
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(index + bar_width / 2, class_names, rotation=45, ha='right', fontsize=10)
    plt.legend()

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_embedding(embeddings, labels, class_names=None, title='t-SNE Embedding',
                   save_path=None, figsize=(12, 10), perplexity=30):
    """
    使用t-SNE算法可视化嵌入空间。

    Args:
        embeddings: 嵌入向量 [n_samples, n_features]
        labels: 标签列表
        class_names: 类别名称
        title: 图表标题
        save_path: 如果提供，保存图表到指定路径
        figsize: 图表大小
        perplexity: t-SNE的perplexity参数
    """
    # 转换为numpy数组
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # 降维到2D
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 绘制散点图
    plt.figure(figsize=figsize)

    # 获取唯一标签
    unique_labels = np.unique(labels)

    # 为每个类别分配颜色
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))

    # 绘制每个类别的点
    for i, label in enumerate(unique_labels):
        idx = labels == label
        label_name = class_names[label] if class_names is not None else f"Class {label}"
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                    c=[cmap(i)], label=label_name, alpha=0.7, s=50)

    # 设置图表属性
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_attention_weights(attention_weights, tokens, title='Attention Weights',
                           save_path=None, figsize=(12, 8)):
    """
    可视化注意力权重。

    Args:
        attention_weights: 注意力权重矩阵 [seq_len, seq_len]
        tokens: 序列中的token列表
        title: 图表标题
        save_path: 如果提供，保存图表到指定路径
        figsize: 图表大小
    """
    # 创建热图
    plt.figure(figsize=figsize)

    ax = sns.heatmap(attention_weights, annot=False, cmap='viridis',
                     xticklabels=tokens, yticklabels=tokens)

    # 设置标签
    plt.ylabel('Query', fontsize=12)
    plt.xlabel('Key', fontsize=12)
    plt.title(title, fontsize=14)

    # 设置刻度标签旋转
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_hierarchy_graph(graph, title='Label Hierarchy Graph', save_path=None, figsize=(15, 10)):
    """
    可视化标签层次图。

    Args:
        graph: networkx图对象
        title: 图表标题
        save_path: 如果提供，保存图表到指定路径
        figsize: 图表大小
    """
    # 确保matplotlib可用
    import matplotlib.pyplot as plt
    import networkx as nx

    plt.figure(figsize=figsize)

    # 获取节点层级
    node_levels = nx.get_node_attributes(graph, 'level')

    # 按层级分组
    level1_nodes = [n for n, d in node_levels.items() if d == 1]
    level2_nodes = [n for n, d in node_levels.items() if d == 2]
    level3_nodes = [n for n, d in node_levels.items() if d == 3]

    # 创建层级布局
    pos = {}

    # L1节点放在顶部
    l1_count = len(level1_nodes)
    for i, node in enumerate(level1_nodes):
        pos[node] = (i - l1_count / 2, 3)

    # L2节点放在中间
    l2_count = len(level2_nodes)
    for i, node in enumerate(level2_nodes):
        pos[node] = (i - l2_count / 4, 2)

    # L3节点放在底部
    l3_count = len(level3_nodes)
    for i, node in enumerate(level3_nodes):
        pos[node] = (i - l3_count / 8, 1)

    # 绘制节点
    nx.draw_networkx_nodes(graph, pos, nodelist=level1_nodes, node_color='skyblue',
                           node_size=200, label='Level 1')
    nx.draw_networkx_nodes(graph, pos, nodelist=level2_nodes, node_color='lightgreen',
                           node_size=150, label='Level 2')
    nx.draw_networkx_nodes(graph, pos, nodelist=level3_nodes, node_color='salmon',
                           node_size=100, label='Level 3')

    # 绘制边
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, arrows=True)

    # 绘制标签（可能会很多，可以选择不显示）
    if len(graph.nodes) < 30:  # 只在节点不多时显示标签
        node_labels = nx.get_node_attributes(graph, 'name')
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8)

    # 设置图表属性
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.legend()

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()
```

### 6. 配置文件 (`configs/config.yaml`)

配置文件可以根据需要调整超参数：

```yaml
# 层次文本分类模型配置

# 模型配置
model:
  bert_model_name: 'bert-base-uncased'  # 使用的预训练语言模型
  max_length: 256  # 输入文本的最大长度
  label_embedding_dim: 128  # 标签嵌入维度
  dropout: 0.1  # Dropout比例
  save_dir: 'D:/Project/tectClass/models'  # 模型保存路径
  save_every: 1  # 每训练多少个epoch保存一次模型

# 训练配置
training:
  batch_size: 32  # 批处理大小
  learning_rate: 0.00002  # 学习率
  num_epochs: 1  # 训练轮数
  weight_decay: 0.01  # 权重衰减
  warmup_ratio: 0.1  # 预热步数比例
  max_grad_norm: 1.0  # 梯度裁剪阈值
  temperature: 0.5  # 对比学习温度参数
  contrastive_loss_weight: 0.5  # 对比学习损失权重
  l2_loss_weight: 1.0  # L2损失权重
  l3_loss_weight: 1.0  # L3损失权重
  seed: 42  # 随机种子
  mask_threshold: 0.3  # 掩码阈值
  adam_epsilon: 1e-8  # Adam优化器epsilon值

# 评估配置
evaluation:
  batch_size: 32  # 评估批处理大小
  analyze_errors: true  # 是否分析错误
  plot_confusion_matrix: true  # 是否绘制混淆矩阵
  plot_distributions: true  # 是否绘制预测分布图

# 数据配置
data:
  train_dir: 'D:/Project/tectClass/data/dataset/test'  # 训练数据目录
  eval_dir: 'data/wiki/eval/'  # 评估数据目录

# 日志配置
logging:
  log_dir: 'logs/'  # 日志保存路径
  log_steps: 100  # 每多少步记录一次日志
```