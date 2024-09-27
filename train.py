import copy
from pathlib import Path
import random
from statistics import mean
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

random_seed = 42  # set random seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

n_way = 3
n_shot = 1
n_query = 20

DEVICE = "cuda"
n_workers = 6
data_name = 'K8000'
from easyfsl.datasets import CUB
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader

n_tasks_per_epoch = 200
n_validation_tasks = 40

train_set = CUB(split="train", training=True, image_size=84)
val_set = CUB(split="val", training=False, image_size=84)

train_sampler = TaskSampler(train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_per_epoch)
val_sampler = TaskSampler(val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks)

train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)
val_loader = DataLoader(
    val_set,
    batch_sampler=val_sampler,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=val_sampler.episodic_collate_fn,
)

from easyfsl.methods import ClipPrototypicalNetworks
from easyfsl.modules import resnet12

# backbone_name = 'resnet10t'swin_s3_small_224
backbone_name = 'mamba'
model_name = 'ClipPrototypicalNetworks'
import timm


# convolutional_network = timm.create_model('resnet10t', pretrained=False)
# convolutional_network = timm.create_model('swin_s3_small_224', pretrained=True)
convolutional_network = resnet12()

# convolutional_network = nn.DataParallel(convolutional_network)  # mamba不能开平行

few_shot_classifier = ClipPrototypicalNetworks(backbone = convolutional_network).to(DEVICE)  # 使用对比原型,加原型修正

from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

LOSS_FUNCTION = nn.CrossEntropyLoss()

n_epochs = 10
scheduler_milestones = [1,4,6,8,10]
scheduler_gamma = 0.1
learning_rate = 1e-4
tb_logs_dir = Path("./logs")

train_optimizer = torch.optim.Adam(few_shot_classifier.parameters(), lr=learning_rate, weight_decay=5e-4)
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(train_optimizer, milestones=scheduler_milestones,
                                                       gamma=scheduler_gamma)

tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))


def training_epoch(model, data_loader: DataLoader, optimizer: Optimizer):
    all_loss = []
    model.train()
    with tqdm(enumerate(data_loader), total=len(data_loader), desc="Training") as tqdm_train:
        for episode_index, (support_images, support_labels, query_images, query_labels, _,) in tqdm_train:
            optimizer.zero_grad()
            model.process_support_set(support_images.to(DEVICE), support_labels.to(DEVICE))
            classification_scores = model(query_images.to(DEVICE))

            loss_l = LOSS_FUNCTION(classification_scores, query_labels.to(DEVICE))
            loss = loss_l + model.contrast_loss  # 交叉损失加对比损失
            loss.backward()
            optimizer.step()
            all_loss.append(loss.item())
            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)


from easyfsl.utils import evaluate
best_state = few_shot_classifier.state_dict()
best_validation_accuracy = 0.0
if __name__ == '__main__':

    print(f'{model_name}-{data_name}-{backbone_name}-{n_way}way-{n_shot}shot')
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        average_loss = training_epoch(few_shot_classifier, train_loader, train_optimizer)
        validation_accuracy = evaluate(few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(few_shot_classifier.state_dict())
            print("Ding ding ding! We found a new best model!")

        tb_writer.add_scalar("Train/loss", average_loss, epoch)
        tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)
        train_scheduler.step()

    few_shot_classifier.load_state_dict(best_state)
    n_test_tasks = 500
    test_set = CUB(split="val", training=False, image_size=84)
    test_sampler = TaskSampler(test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks)
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    accuracy = evaluate(few_shot_classifier, test_loader, device=DEVICE)
    print(f"Average accuracy : {(100 * accuracy):.2f} %")

