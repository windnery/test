from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalTextModel(nn.Module):
    def __init__(self, bert_model_name, num_labels, label_embedding_dim, label_graph):
        super(HierarchicalTextModel, self).__init__()
        self.encoder = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # 标签嵌入
        self.label_embedding = nn.Embedding(num_embeddings=len(label_graph.nodes), embedding_dim=label_embedding_dim)
        # Graphormer模块（简化版）
        self.graph_encoder = Graphormer(hidden_size=label_embedding_dim)
        self.label_graph = label_graph

        # 投影头用于对比学习
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # 分类器
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, positive_input_ids=None, positive_attention_mask=None):
        # 编码原始文本
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token

        # 编码标签层次结构
        label_features = self.get_label_features()
        # 后加
        linear_layer = nn.Linear(128, 768).to(label_features.device)
        label_features = linear_layer(label_features)
        # 组合文本表示和标签表示
        combined_features = sequence_output + label_features

        # 分类输出
        logits = self.classifier(combined_features)

        # 对比学习的表示
        z_i = self.projection_head(combined_features)

        if positive_input_ids is not None and positive_attention_mask is not None:
            # 编码正样本
            positive_outputs = self.encoder(input_ids=positive_input_ids, attention_mask=positive_attention_mask)
            positive_sequence_output = positive_outputs.last_hidden_state[:, 0, :]
            positive_combined_features = positive_sequence_output + label_features
            z_j = self.projection_head(positive_combined_features)
            return logits, z_i, z_j
        else:
            return logits, z_i

    def get_label_features(self):
        # 获取标签嵌入
        label_embeddings = self.label_embedding.weight  # [num_labels, label_embedding_dim]
        # 通过图编码器编码标签特征
        encoded_label_features = self.graph_encoder(label_embeddings, self.label_graph)
        # 将标签特征汇总（简化处理）
        label_features = torch.mean(encoded_label_features, dim=0)
        return label_features

class Graphormer(nn.Module):
    def __init__(self, hidden_size):
        super(Graphormer, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

    def forward(self, label_embeddings, label_graph):
        # label_embeddings: [num_labels, hidden_size]
        # 需要根据label_graph构建mask或edge属性，这里简化处理
        encoded_labels = self.encoder(label_embeddings.unsqueeze(1)).squeeze(1)
        return encoded_labels