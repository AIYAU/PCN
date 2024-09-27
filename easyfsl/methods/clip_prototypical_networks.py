from typing import Optional
import torch.nn.functional as F
import torch
from torch import Tensor, nn

from easyfsl.methods.utils import compute_prototypes


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

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return out1, out2


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = NonSharedSiameseNetwork(input_dim=640, hidden_dim=256, output_dim=768)

    def forward(self, feature):
        f1, f2 = self.model(feature)
        # print(f"{f1.shape, f2.shape}")
        return f1 @ f2.T


class ClipPrototypicalNetworks(nn.Module):
    def __init__(
            self,
            backbone: Optional[nn.Module] = None,
            use_softmax: bool = True,
            feature_centering: Optional[Tensor] = None,
            feature_normalization: Optional[float] = None,
            use_rectify: bool = False
    ):
        """
        Initialize the ClipPrototypicalNetworks
        Args:
            backbone: the feature extractor used by the method. Must output a tensor of the
                appropriate shape (depending on the method).
                If None is passed, the backbone will be initialized as nn.Identity().
            use_softmax: whether to return predictions as soft probabilities
            feature_centering: a features vector on which to center all computed features.
                If None is passed, no centering is performed.
            feature_normalization: a value by which to normalize all computed features after centering.
                It is used as the p argument in torch.nn.functional.normalize().
                If None is passed, no normalization is performed.
        """
        super().__init__()

        self.backbone = backbone if backbone is not None else nn.Identity()
        self.use_softmax = use_softmax
        self.use_rectify = use_rectify

        self.prototypes = torch.tensor(())  # 用来存储原型的
        self.support_features = torch.tensor(())
        self.support_labels = torch.tensor(())

        self.feature_centering = (
            feature_centering if feature_centering is not None else torch.tensor(0)
        )
        self.feature_normalization = feature_normalization
        self.clip = CLIP()
        self.contrast_loss = torch.tensor(())
        self.cs_wt = torch.nn.Parameter(torch.normal(mean=.1, std=1e-4, size=[], dtype=torch.float, device='cuda'),requires_grad=True)

    def process_support_set(self, support_images: Tensor, support_labels: Tensor, ):
        """
        Harness information from the support set, so that query labels can later be predicted using a forward call.
        The default behaviour shared by most few-shot classifiers is to compute prototypes and store the support set.
        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    def compute_features(self, images: Tensor) -> Tensor:
        """
        Compute features from images and perform centering and normalization.
        Args:
            images: images of shape (n_images, **image_shape)
        Returns:
            features of shape (n_images, feature_dimension)
        """
        original_features = self.backbone(images)
        centered_features = original_features - self.feature_centering
        if self.feature_normalization is not None:
            return nn.functional.normalize(centered_features, p=self.feature_normalization, dim=1)
        return centered_features

    def gaussian_distribution(self,center: int, num_classes: int, sigma: float) -> Tensor:
        """
        Generates a Gaussian distribution over the number of classes, centered on the given center index.
        Args:
            center: The class index to be the mean (center) of the Gaussian distribution.
            num_classes: Total number of classes.
            sigma: The standard deviation of the Gaussian distribution.
        Returns:
            A tensor of size (num_classes,) representing the Gaussian distribution over classes.
        """
        # Create a range of class indices and compute Gaussian distribution
        class_indices = torch.arange(num_classes,device='cuda')
        gaussian_probs = torch.exp(-0.5 * ((class_indices - center) ** 2) / (sigma ** 2))
        
        # Normalize the Gaussian to ensure it sums to 1
        return gaussian_probs / gaussian_probs.sum()

    def softmax_if_specified(self, output: Tensor, temperature: float = 1, sigma: float = 1.2) -> Tensor:
            """
            If the option is chosen when the classifier is initialized, we perform a softmax on the
            output and then apply Gaussian smoothing to return soft probabilities.
            Args:
                output: output of the forward method of shape (n_query, n_classes)
                temperature: temperature of the softmax
                sigma: standard deviation for Gaussian smoothing
            Returns:
                output as it was, or output as soft probabilities with Gaussian smoothing, 
                of shape (n_query, n_classes)
            """
            if not self.use_softmax:
                return output
            
            # Apply temperature scaling and softmax
            output = (output / temperature).softmax(dim=-1)
            
            n_query, n_classes = output.shape
            
            # Apply Gaussian distribution for each output row
            gaussian_smoothed_output = torch.zeros_like(output)
            
            for i in range(n_query):
                # Get the index of the class with the highest probability (argmax)
                predicted_class = output[i].argmax().item()
                
                # Generate a Gaussian distribution centered on the predicted class
                gaussian_probs = self.gaussian_distribution(predicted_class, n_classes, sigma)
                
                # Multiply the softmax probabilities with the Gaussian distribution
                gaussian_smoothed_output[i] = output[i] * gaussian_probs
            
            return gaussian_smoothed_output
    # def softmax_if_specified(self, output: Tensor, temperature: float = 1.2) -> Tensor:
    #     """
    #     If the option is chosen when the classifier is initialized, we perform a softmax on the
    #     output in order to return soft probabilities.
    #     Args:
    #         output: output of the forward method of shape (n_query, n_classes)
    #         temperature: temperature of the softmax
    #     Returns:
    #         output as it was, or output as soft probabilities, of shape (n_query, n_classes)
    #     """
    #     return (temperature * output).softmax(-1) if self.use_softmax else output

    def l2_distance_to_prototypes(self, samples: Tensor) -> Tensor:
        """
        Compute prediction logits from their euclidean distance to support set prototypes.
        Args:
            samples: features of the items to classify of shape (n_samples, feature_dimension)
        Returns:
            prediction logits of shape (n_samples, n_classes)
        """
        return -torch.cdist(samples, self.prototypes)

    def cosine_distance_to_prototypes(self, samples) -> Tensor:
        """
        Compute prediction logits from their cosine distance to support set prototypes.
        Args:
            samples: features of the items to classify of shape (n_samples, feature_dimension)
        Returns:
            prediction logits of shape (n_samples, n_classes)
        """
        return (
                nn.functional.normalize(samples, dim=1)
                @ nn.functional.normalize(self.prototypes, dim=1).T
        )

    def compute_prototypes_and_store_support_set(self, support_images: Tensor, support_labels: Tensor, ):
        """
        Extract support features, compute prototypes, and store support labels, features, and prototypes.
        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
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

    def rectify_prototypes(self, query_features: Tensor):  # pylint: disable=not-callable
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

    def forward(self, query_images: Tensor, ) -> Tensor:

        query_features = self.compute_features(query_images)  # 25 = 5way * 5query
        if self.use_rectify:
            self.rectify_prototypes(query_features = query_features)

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
