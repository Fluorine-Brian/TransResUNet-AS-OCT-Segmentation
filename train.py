import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 确保 model.py, utils.py, metrics.py 文件存在且包含所需的类和函数
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
from model import TResUnet
from metrics import DiceLoss, DiceBCELoss


# 修改 Dataset 类，使其可以接受一个文件列表作为输入
class OCTDataset(Dataset):
    def __init__(self, base_folder, image_subfolder, mask_subfolder, size, transform=None, filenames_list=None):
        """
        自定义数据集类，用于加载OCT图像及其对应的单个掩码。

        Args:
            base_folder (str): 数据集根目录 (例如 'path/to/dataset_split/train')
            image_subfolder (str): 图像子文件夹名称 (例如 'images')
            mask_subfolder (str): 掩码子文件夹名称 (例如 'anterior_chamber_masks')
            size (tuple): 图像和掩码将被resize到的尺寸 (width, height)
            transform (albumentations.Compose, optional): 数据增强转换. Defaults to None.
            filenames_list (list, optional): 如果提供，则使用此列表中的文件名加载数据，
                                            否则从 base_folder/image_subfolder 中 glob 所有文件。
        """
        super().__init__()

        self.base_folder = base_folder
        self.image_subfolder = image_subfolder
        self.mask_subfolder = mask_subfolder

        if filenames_list is not None:
            # 使用提供的文件名列表构建完整路径
            self.image_paths = [os.path.join(base_folder, image_subfolder, f) for f in filenames_list]
            # 假设掩码文件名与原始图像文件名（不含扩展名）一致，且掩码扩展名为 .png
            self.mask_paths = [os.path.join(base_folder, mask_subfolder, os.path.splitext(f)[0] + '.png') for f in
                               filenames_list]
        else:
            # 如果未提供文件名列表，则从文件夹中 glob 所有文件 (原始行为)
            self.image_paths = sorted(glob(os.path.join(base_folder, image_subfolder, "*")))
            self.mask_paths = []
            for img_path in self.image_paths:
                img_filename = os.path.basename(img_path)
                base_filename = os.path.splitext(img_filename)[0]
                mask_filename = base_filename + '.png'
                mask_path = os.path.join(base_folder, mask_subfolder, mask_filename)
                self.mask_paths.append(mask_path)

        self.size = size
        self.transform = transform
        self.n_samples = len(self.image_paths)

        # 检查图像和掩码数量是否匹配
        if len(self.image_paths) != len(self.mask_paths):
            print(f"警告: 图像数量 ({len(self.image_paths)}) 与掩码数量 ({len(self.mask_paths)}) 不匹配。")
            # 注意：如果 filenames_list 导致不匹配，这里会警告。确保你的文件名列表是基于原始图像文件生成的。

    def __getitem__(self, index):
        """ Image """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # 读取图像 (灰度)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"错误: 无法读取图像文件 {image_path}")
            # 返回一个占位符，或者跳过，这里返回零数组作为示例
            image = np.zeros((self.size[1], self.size[0]),
                             dtype=np.uint8)  # 注意cv2尺寸是(width, height)，numpy是(height, width)
            mask = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        else:
            # 读取掩码 (灰度)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"警告: 无法读取掩码文件 {mask_path}，使用全黑掩码代替。")
                mask = np.zeros_like(image, dtype=np.uint8)  # 使用与图像相同尺寸的零数组

        # 应用数据增强 (Albumentations 需要彩色图像作为输入，即使是灰度图)
        # 将灰度图像转换为3通道，以便Albumentations处理
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if self.transform is not None:
            # Albumentations 返回字典
            augmentations = self.transform(image=image_rgb, mask=mask)
            image_rgb = augmentations["image"]
            mask = augmentations["mask"]

        # Resize 图像和掩码
        # cv2.resize 期望 (width, height)
        image_resized = cv2.resize(image_rgb, self.size)
        mask_resized = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)  # 掩码使用最近邻插值，避免引入中间值

        # 转换为 PyTorch Tensor 格式 (C, H, W) 并归一化
        # 图像: (H, W, C) -> (C, H, W), 归一化到 [0, 1]
        image_tensor = np.transpose(image_resized, (2, 0, 1)).astype(np.float32) / 255.0

        # 掩码: (H, W) -> (1, H, W), 归一化到 [0, 1] (二值掩码 0 或 1)
        # 确保掩码是二值的 (0 或 1)，即使resize可能引入中间值
        mask_tensor = (mask_resized > 127).astype(np.float32)  # 阈值化为二值
        mask_tensor = np.expand_dims(mask_tensor, axis=0)  # 添加通道维度

        return image_tensor, mask_tensor

    def __len__(self):
        return self.n_samples


# 原有的 train 和 evaluate 函数保持不变，因为它们处理的是 PyTorch Tensor，
# 并且模型输出和掩码都是单通道的，适合二分类。
# from train import train, evaluate # 如果 train 和 evaluate 在另一个文件，需要导入
# 这里假设 train 和 evaluate 函数就在当前文件中定义。


def train(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        # 将预测结果通过 sigmoid 转换为概率，然后阈值化为二值掩码 (0或1)
        y_pred_binary = (torch.sigmoid(y_pred) > 0.5).float()
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y, y_pred_binary):  # 使用二值化后的预测结果计算指标
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss /= len(loader)
    epoch_jac /= len(loader)
    epoch_f1 /= len(loader)
    epoch_recall /= len(loader)
    epoch_precision /= len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            """ Calculate the metrics """
            # 将预测结果通过 sigmoid 转换为概率，然后阈值化为二值掩码 (0或1)
            y_pred_binary = (torch.sigmoid(y_pred) > 0.5).float()
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y, y_pred_binary):  # 使用二值化后的预测结果计算指标
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss /= len(loader)
        epoch_jac /= len(loader)
        epoch_f1 /= len(loader)
        epoch_recall /= len(loader)
        epoch_precision /= len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    # 输出文件和日志将保存在这里
    output_files_base_dir = "files_oct_segmentation"
    create_dir(output_files_base_dir)

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print(f"脚本开始时间: {datetime_object}")

    """ Hyperparameters """
    image_size = 256  # 模型输入尺寸
    size = (image_size, image_size)  # (width, height)
    batch_size = 16
    num_epochs = 500  # 可以根据需要调整
    lr = 1e-4
    early_stopping_patience = 50  # 如果验证集F1连续50个epoch没有提升，则停止训练

    # 数据集根路径 (包含 train 和 test 文件夹)
    dataset_root_path = r"C:/srp_OCT/segmentation_mask/segmentation_mask_dataset"  # 替换为你的数据集划分后的根路径

    # 定义需要训练的掩码类型及其对应的子文件夹名称
    mask_types_to_train = {
        'anterior_chamber': 'qf_masks',
        'lens': 'jzt_masks',
        'cornea': 'jm_masks',
        'iris': 'hm_masks'
    }
    image_subfolder_name = 'images'  # 原始图像子文件夹名称

    # 数据增强转换
    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
        # 可以根据需要添加更多增强
    ])

    """ Device """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 循环训练每种掩码类型 ---
    for mask_type, mask_subfolder_name in mask_types_to_train.items():
        print(f"\n--- 开始训练 {mask_type} 的分割模型 ---")

        # 训练日志文件路径 (为每种掩码类型创建独立的日志)
        train_log_path = os.path.join(output_files_base_dir, f"train_log_{mask_type}.txt")
        # 使用追加模式打开日志文件，如果不存在则创建
        train_log_file = open(train_log_path, "a")
        print_and_save(train_log_file, f"\n--- 训练 {mask_type} 开始 ---")
        print_and_save(train_log_file, f"训练开始时间: {datetime_object}")
        print_and_save(train_log_file, f"训练掩码类型: {mask_type}\n")
        print_and_save(train_log_file, f"图像尺寸: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n")
        print_and_save(train_log_file, f"Early Stopping Patience: {early_stopping_patience}\n")
        print_and_save(train_log_file, "-" * 30 + "\n")

        # 检查点保存路径 (为每种掩码类型创建独立的检查点)
        checkpoint_path = os.path.join(output_files_base_dir, f"checkpoint_{mask_type}.pth")

        # --- 数据集路径 ---
        # 现在只使用 train 文件夹的数据
        all_train_data_base_folder = os.path.join(dataset_root_path, 'train')

        # --- 获取 train 文件夹中所有图像的文件名 ---
        all_image_filenames = [f for f in os.listdir(os.path.join(all_train_data_base_folder, image_subfolder_name))
                               if os.path.isfile(os.path.join(all_train_data_base_folder, image_subfolder_name, f))
                               and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

        if not all_image_filenames:
            print_and_save(train_log_file,
                           f"错误: 在 {all_train_data_base_folder}/{image_subfolder_name} 中没有找到图像文件，跳过 {mask_type} 的训练。")
            train_log_file.close()
            continue

        # --- 随机打乱文件名并划分新的训练集和验证集 (3:1) ---
        random.shuffle(all_image_filenames)
        num_total_data = len(all_image_filenames)
        num_new_train = int(num_total_data * 0.75)  # 3/4 用于训练
        num_new_val = num_total_data - num_new_train  # 剩余用于验证

        new_train_filenames = all_image_filenames[:num_new_train]
        new_val_filenames = all_image_filenames[num_new_train:]

        # --- 创建数据集实例，使用划分后的文件名列表 ---
        # 注意：base_folder 仍然指向原始的 train 文件夹
        train_dataset = OCTDataset(all_train_data_base_folder, image_subfolder_name, mask_subfolder_name, size,
                                   transform=transform, filenames_list=new_train_filenames)
        valid_dataset = OCTDataset(all_train_data_base_folder, image_subfolder_name, mask_subfolder_name, size,
                                   transform=None, filenames_list=new_val_filenames)  # 验证集不进行数据增强

        # 检查划分后的数据集是否为空
        if len(train_dataset) == 0:
            print_and_save(train_log_file, f"错误: 划分后的训练集为空，跳过 {mask_type} 的训练。")
            train_log_file.close()
            continue
        if len(valid_dataset) == 0:
            print_and_save(train_log_file, f"警告: 划分后的验证集为空，将无法进行验证和早停。")

        data_str = f"数据集大小 (从 {os.path.basename(all_train_data_base_folder)} 划分):\n训练集: {len(train_dataset)} - 验证集: {len(valid_dataset)}\n"
        print_and_save(train_log_file, data_str)

        # 创建 DataLoader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2  # 根据你的机器性能调整 num_workers
        )

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2  # 根据你的机器性能调整 num_workers
        )

        """ Model """
        model = TResUnet()  # 每次训练新的掩码类型时重新实例化模型
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # 使用验证集损失作为 ReduceLROnPlateau 的监控指标
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
        loss_fn = DiceBCELoss()  # 适用于二分类
        loss_name = "BCE Dice Loss"
        data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
        print_and_save(train_log_file, data_str)

        """ Training the model """
        best_valid_metrics = 0.0  # 监控指标，这里使用 F1 Score
        early_stopping_count = 0

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)

            # 只有当验证集不为空时才进行评估和早停判断
            if len(valid_dataset) > 0:
                valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
                scheduler.step(valid_loss)  # 使用验证集损失更新学习率

                # 监控验证集 F1 Score
                if valid_metrics[1] > best_valid_metrics:
                    data_str = f"Epoch {epoch + 1:02}: Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
                    print_and_save(train_log_file, data_str)

                    best_valid_metrics = valid_metrics[1]
                    torch.save(model.state_dict(), checkpoint_path)
                    early_stopping_count = 0

                else:
                    early_stopping_count += 1
                    print_and_save(train_log_file,
                                   f"Epoch {epoch + 1:02}: Valid F1 did not improve. Early stopping count: {early_stopping_count}/{early_stopping_patience}")

            else:  # 如果划分后的验证集为空，则不进行验证和早停
                valid_loss = float('inf')  # 标记验证损失为无穷大
                valid_metrics = [0.0] * 4  # 标记验证指标为0
                scheduler.step(train_loss)  # 如果没有验证集，可以使用训练损失更新学习率 (不推荐，但为了代码运行)
                print_and_save(train_log_file, f"Epoch {epoch + 1:02}: 划分后的验证集为空，跳过验证和早停检查。")

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
            data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
            data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
            print_and_save(train_log_file, data_str)

            if len(valid_dataset) > 0 and early_stopping_count >= early_stopping_patience:
                data_str = f"Early stopping triggered: validation F1 stops improving from last {early_stopping_patience} epochs.\n"
                print_and_save(train_log_file, data_str)
                break

        print_and_save(train_log_file, f"--- {mask_type} 模型训练结束 ---")
        train_log_file.close()  # 关闭日志文件

    print("\n所有掩码类型的模型训练完成。")
