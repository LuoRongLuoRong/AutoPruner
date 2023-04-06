from tqdm import tqdm
import numpy as np
from src.finetune.dataset import CallGraphDataset
from torch.utils.data import DataLoader
from src.finetune.model import BERT
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torch
import argparse
from src.utils.utils import Logger, AverageMeter, evaluation_metrics, read_config_file
import os
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练集的参数
TRAIN_PARAMS = {'batch_size': 15, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': 10, 'shuffle': False, 'num_workers': 8}

# 日志记录器
logger = Logger()

# dataloader：包含训练数据的数据加载器，用于批量处理数据。
# model：需要训练的模型。
# mean_loss：平均损失值的计算器，用于记录每个 epoch 的平均损失值。
# loss_fn：损失函数，用于计算模型的损失。
# optimizer：优化器，用于更新模型参数。
# cfx_matrix：混淆矩阵，用于计算模型的精确度、召回率和 F1 值。
def train(dataloader, model, mean_loss, loss_fn, optimizer, cfx_matrix):
    # 调用 model.train()，以确保模型处于训练模式。
    model.train()
    # 用于在 Python 中显示进度条
    loop=tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    # 遍历 dataloader 中的每个 batch
    for idx, batch in loop:
        # 将数据移动到指定的设备（例如 GPU）
        code_ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        label = batch['label'].to(device)
        # 将数据输入模型，得到输出值 output
        output, _=model(
                ids=code_ids,
                mask=mask)

        # 计算输出值和真实标签之间的损失
        loss = loss_fn(output, label)

        num_samples = output.shape[0]
        # 用平均损失值计算器 mean_loss 计算当前 batch 的平均损失值
        mean_loss.update(loss.item(), n=num_samples)
        
        # 将输出值进行 softmax 操作
        output = F.softmax(output)
        # 将输出值转换为 numpy 数组
        output = output.detach().cpu().numpy()[:, 1]
        # 用阈值 0.5 将输出值二元化，并将其与真实标签进行比较，以计算混淆矩阵、精确度、召回率和 F1 值。
        pred = np.where(output >= 0.5, 1, 0)
        # 真实标签
        label = label.detach().cpu().numpy()
        
        # 计算混淆矩阵、精确度、召回率和 F1 值
        cfx_matrix, precision, recall, f1 = evaluation_metrics(label, pred, cfx_matrix)

        # 将 混淆矩阵、精确度、召回率和 F1 值 记录到 logger 中
        logger.log("Iter {}: Loss {}, Precision {}, Recall {}, F1 {}".format(idx, mean_loss.item(), precision, recall, f1))
        # 使用 tqdm 库更新进度条
        loop.set_postfix(loss=mean_loss.item(), pre=precision, rec=recall, f1 = f1)

        # 在更新模型参数之前，使用 optimizer.zero_grad() 将模型的梯度归零。
        optimizer.zero_grad()
        # 计算模型的梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
    
    # 返回更新后的模型和最终的混淆矩阵
    return model, cfx_matrix

def do_test(dataloader, model):
    model.eval()
    cfx_matrix = np.array([[0, 0],
                           [0, 0]])
    loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
    for batch, dl in loop:
        ids=dl['ids'].to(device)
        mask= dl['mask'].to(device)
        label=dl['label'].to(device)
        output, _=model(
                ids=ids,
                mask=mask)
        
        output = F.softmax(output)
        output = output.detach().cpu().numpy()[:, 1]
        pred = np.where(output >= 0.5, 1, 0)
        label = label.detach().cpu().numpy()
        
        cfx_matrix, precision, recall, f1 = evaluation_metrics(label, pred, cfx_matrix)
        loop.set_postfix(pre=precision, rec=recall, f1 = f1)
        
    (tn, fp), (fn, tp) = cfx_matrix
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2*precision*recall/(precision + recall)
    logger.log("[EVAL] Iter {}, Precision {}, Recall {}, F1 {}".format(batch, precision, recall, f1))

# 训练模型，并保存模型参数，以便后续测试使用
def do_train(epochs, train_loader, test_loader, model, loss_fn, optimizer, learned_model_dir):
    cfx_matrix = np.array([[0, 0],
                           [0, 0]])
    mean_loss = AverageMeter()
    for epoch in range(epochs):
        logger.log("Start training at epoch {} ...".format(epoch))
        model, cfx_matrix = train(train_loader, model, mean_loss, loss_fn, optimizer, cfx_matrix)
        
        logger.log("Saving model ...")
        torch.save(model.state_dict(), os.path.join(learned_model_dir, "model_epoch{}.pth".format(epoch)))

        logger.log("Evaluating ...")
        do_test(test_loader, model)
    
    # 将训练好的模型保存到本地
    torch.save(model.state_dict(), os.path.join(learned_model_dir, "model.pth"))
    logger.log("Done !!!")


# 用于解析命令行参数
def get_args():
    parser = argparse.ArgumentParser()
    # 指定模型的配置文件路径、模型文件路径和模式（训练或测试）
    parser.add_argument("--config_path", type=str, default="config/wala.config") 
    parser.add_argument("--model_path", type=str, default="../replication_package/model/finetuned_model/model.pth", help="Path to checkpoint (for test only)") 
    parser.add_argument("--mode", type=str, default="train") 
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = get_args()
    # 从指定路径读取配置文件，并将配置参数存储在 config 变量中
    config = read_config_file(args.config_path)
    print("Running on config {}".format(args.config_path))
    print("Mode: {}".format(args.mode))
    
    # 选择是训练还是测试模式
    mode = args.mode
    learned_model_dir = config["LEARNED_MODEL_DIR"]


    # 初始化训练和测试数据集
    train_dataset= CallGraphDataset(config, "train")
    test_dataset= CallGraphDataset(config, "test")

    print("Dataset have {} train samples and {} test samples".format(len(train_dataset), len(test_dataset)))

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, **TRAIN_PARAMS)
    test_loader = DataLoader(test_dataset, **TEST_PARAMS)

    # 定义模型
    model=BERT()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer= optim.Adam(model.parameters(),lr= 0.00001)

    # 训练模式
    if mode == "train":
        # 进行训练
        do_train(2, train_loader, test_loader, model, loss_fn, optimizer, learned_model_dir)
    # 测试模式
    elif mode == "test":
        # 加载已训练的模型参数
        model.load_state_dict(torch.load(args.model_path))
        # 进行测试
        do_test(test_loader, model)
    else:
        raise NotImplemented

if __name__ == '__main__':
    main()
