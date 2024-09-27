from typing import Optional
import torch.nn.functional as F
import torch
from torch import Tensor, nn

from easyfsl.methods.utils import compute_prototypes

from transformers import GPT2Tokenizer, GPT2Model

from transformers import BertTokenizer, BertModel

# 定义将长度为640的向量转换为BERT输入格式的编码器
class VectorBecoder(nn.Module):
    def __init__(self, input_dim=768, bert_dim=768):
        super(VectorBecoder, self).__init__()
        # 这块可以自由扩展个性化定义
        self.fc = nn.Linear(input_dim, bert_dim)
        # 加载预训练的BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x):
        # 编码输入向量
        encoded_input = self.fc(x)  # 首先进行维度变换
        # 将编码后的输入转换为BERT模型的输入格式
        # BERT模型的输入要求是 (batch_size, sequence_length, hidden_size)
        # 这里我们将sequence_length设为1，并将encoded_input扩展到此维度
        encoded_input = encoded_input.unsqueeze(1)
        # 进行推理，得到BERT的输出
        with torch.no_grad():
            bert_output = self.bert_model(inputs_embeds=encoded_input)
        return bert_output.last_hidden_state.squeeze(1)

# 定义将长度为640的向量转换为GPT-2输入格式的编码器
# Mapping 映射
class VectorDecoder(nn.Module):
    def __init__(self, input_dim=768, gpt2_dim=768):
        super(VectorDecoder, self).__init__()
        # 这块可以自由扩展个性化定义
        self.fc = nn.Linear(input_dim, gpt2_dim)
        # 加载预训练的GPT-2模型和分词器
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', mirror="tuna")
        self.gpt2_model = GPT2Model.from_pretrained('gpt2', mirror="tuna")

    def forward(self, x):
        # 编码输入向量
        encoded_input = self.fc(x)  # 首先进行维度变换
        # 将编码后的输入转换为GPT-2模型的输入格式
        # GPT-2模型的输入要求是 (batch_size, sequence_length, hidden_size)
        # 这里我们将sequence_length设为1，并将encoded_input扩展到此维度
        encoded_input = encoded_input.unsqueeze(1)
        # 进行推理，得到GPT-2的输出
        with torch.no_grad():
            gpt2_output = self.gpt2_model(inputs_embeds=encoded_input)
        return gpt2_output.last_hidden_state.squeeze(1)


class NonSharedSiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        非共享孪生网络，将特征向量分裂
        '''
        super(NonSharedSiameseNetwork, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # self.textencoder = VectorDecoder().cuda()# GPT
        self.textencoder = VectorBecoder().cuda()# BERT

    def forward(self, x):
        # print(x.shape,'====')
        images = self.branch1(x)
        text = self.branch2(x)
        # text = text + self.textencoder(text) #GPT2
        text = self.textencoder(text) #Bert
        # print(images.shape, text.shape, '=======')
        return images, text


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = NonSharedSiameseNetwork(input_dim=1000, hidden_dim=256, output_dim=768)

    def forward(self, feature):
        f1, f2 = self.model(feature)
        # print(f"{f1.shape, f2.shape}")
        return f1 @ f2.T


class ClipGPT2PrototypicalNetworks(nn.Module):
    def __init__(
            self,
            backbone: Optional[nn.Module] = None,
            use_softmax: bool = False,
            feature_centering: Optional[Tensor] = None,
            feature_normalization: Optional[float] = None,
    ):

        super().__init__()

        self.backbone = backbone if backbone is not None else nn.Identity()
        self.use_softmax = use_softmax

        self.prototypes = torch.tensor(())  # 用来存储原型的
        self.support_features = torch.tensor(())
        self.support_labels = torch.tensor(())

        self.feature_centering = (
            feature_centering if feature_centering is not None else torch.tensor(0)
        )
        self.feature_normalization = feature_normalization
        self.clip = CLIP()
        self.contrast_loss = torch.tensor(())
        self.cs_wt = torch.nn.Parameter(torch.normal(mean=.1, std=1e-4, size=[], dtype=torch.float, device='cuda'),
                                        requires_grad=True)


    def rectify_prototypes(self, query_features: Tensor):  # 原型修正
        """
        Updates prototypes with label propagation and feature shifting.
        Args:
            query_features: query features of shape (n_query, feature_dimension)
        """
        n_classes = self.support_labels.unique().size(0)
        one_hot_support_labels = nn.functional.one_hot(self.support_labels, n_classes)
        average_support_query_shift = self.support_features.mean(0, keepdim=True) - query_features.mean(0, keepdim=True)
        query_features = query_features + average_support_query_shift
        support_logits = self.cosine_distance_to_prototypes(self.support_features).exp()
        query_logits = self.cosine_distance_to_prototypes(query_features).exp()
        one_hot_query_prediction = nn.functional.one_hot(query_logits.argmax(-1), n_classes)
        normalization_vector = ((one_hot_support_labels * support_logits).sum(0)+ (one_hot_query_prediction * query_logits).sum(0)).unsqueeze(0)  # [1, n_classes]
        support_reweighting = (one_hot_support_labels * support_logits) / normalization_vector  # [n_support, n_classes]
        query_reweighting = (one_hot_query_prediction * query_logits) / normalization_vector  # [n_query, n_classes]
        self.prototypes = (support_reweighting * one_hot_support_labels).t().matmul(self.support_features) + (query_reweighting * one_hot_query_prediction).t().matmul(query_features)


    def process_support_set(self, support_images: Tensor, support_labels: Tensor, ):
        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    def compute_features(self, images: Tensor) -> Tensor:

        original_features = self.backbone(images)
        centered_features = original_features - self.feature_centering
        if self.feature_normalization is not None:
            return nn.functional.normalize(centered_features, p=self.feature_normalization, dim=1)
        return centered_features

    def softmax_if_specified(self, output: Tensor, temperature: float = 1.0) -> Tensor:
        return (temperature * output).softmax(-1) if self.use_softmax else output

    def l2_distance_to_prototypes(self, samples: Tensor) -> Tensor:
        return -torch.cdist(samples, self.prototypes)

    def cosine_distance_to_prototypes(self, samples) -> Tensor:
        return (
                nn.functional.normalize(samples, dim=1)
                @ nn.functional.normalize(self.prototypes, dim=1).T
        )

    def compute_prototypes_and_store_support_set(self, support_images: Tensor, support_labels: Tensor, ):

        self.support_labels = support_labels  # 保存支持集标签
        self.support_features = self.compute_features(support_images)  # 利用主干提取特征
        self._raise_error_if_features_are_multi_dimensional(self.support_features)  # 确保特征是1维的
        self.prototypes = compute_prototypes(self.support_features, support_labels)  # 计算原型

    @staticmethod

    def _raise_error_if_features_are_multi_dimensional(features: Tensor):
        if len(features.shape) != 2:
            raise ValueError(
                "Illegal backbone or feature shape. "
                "Expected output for an image is a 1-dim tensor."
            )

    def forward(self, query_images: Tensor, ) -> Tensor:

        query_features = self.compute_features(query_images)  # 25 = 5 way * 5query

        self._raise_error_if_features_are_multi_dimensional(query_features)
        # 计算对比损失
        logits = self.clip(self.prototypes)
        # print(logits.shape)
        # There 3 is the number of classes
        targets = torch.arange(0, 3).to('cuda')
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.permute(1, 0), targets)
        self.contrast_loss = (loss_i + loss_t) / 2

        # Compute the euclidean distance from queries to prototypes
        scores = self.l2_distance_to_prototypes(query_features)
        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False
