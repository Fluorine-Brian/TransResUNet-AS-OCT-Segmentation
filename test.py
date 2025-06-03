import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from operator import add  # 用于累加指标
from tqdm import tqdm  # 用于显示进度条 (需要安装: pip install tqdm)

# 确保 model.py, utils.py, metrics.py 文件存在且包含所需的类和函数
from model import TResUnet  # 确保 model.py 文件存在且包含 TResUnet 类
# from metrics import DiceLoss, DiceBCELoss # 评估时通常不需要损失函数
from utils import calculate_metrics, create_dir  # 确保 utils.py 文件存在且包含 calculate_metrics 和 create_dir 函数


# 沿用训练脚本中的 Dataset 类
class OCTDataset(Dataset):
    def __init__(self, base_folder, image_subfolder, mask_subfolder, size):
        """
        自定义数据集类，用于加载OCT图像及其对应的单个掩码。
        用于评估时，返回图像和掩码的路径以及处理后的 tensor。

        Args:
            base_folder (str): 数据集根目录 (例如 'path/to/dataset_split/test')
            image_subfolder (str): 图像子文件夹名称 (例如 'images')
            mask_subfolder (str): 掩码子文件夹名称 (例如 'anterior_chamber_masks')
            size (tuple): 图像和掩码将被resize到的尺寸 (width, height)
        """
        super().__init__()

        # 使用 glob 获取图像路径，并按文件名排序以确保与掩码对应
        self.image_paths = sorted(glob(os.path.join(base_folder, image_subfolder, "*")))
        self.mask_paths = []

        # 假设掩码文件名与原始图像文件名（不含扩展名）一致，且掩码扩展名为 .png
        for img_path in self.image_paths:
            img_filename = os.path.basename(img_path)
            base_filename = os.path.splitext(img_filename)[0]
            mask_filename = base_filename + '.png'  # 掩码扩展名为 .png
            mask_path = os.path.join(base_folder, mask_subfolder, mask_filename)
            self.mask_paths.append(mask_path)

        self.size = size
        self.n_samples = len(self.image_paths)

        # 检查图像和掩码数量是否匹配
        if len(self.image_paths) != len(self.mask_paths):
            print(
                f"警告: 图像数量 ({len(self.image_paths)}) 与掩码数量 ({len(self.mask_paths)}) 不匹配在文件夹 {base_folder}/{mask_subfolder}")
            # 如果数量不匹配，可以进一步检查哪些文件缺失，或者直接过滤掉不匹配的对
            # 为了简单，这里只打印警告，假设文件是按顺序对应的

    def __getitem__(self, index):
        """ Image """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # 读取图像 (灰度)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"错误: 无法读取图像文件 {image_path}")
            # 返回一个占位符，或者跳过，这里返回零数组作为示例
            # 注意cv2尺寸是(width, height)，numpy是(height, width)
            image = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
            mask = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        else:
            # 读取掩码 (灰度)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"警告: 无法读取掩码文件 {mask_path}，使用全黑掩码代替。")
                mask = np.zeros_like(image, dtype=np.uint8)  # 使用与图像相同尺寸的零数组

        # 将灰度图像转换为3通道，以匹配模型输入的期望
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

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

        # 返回处理后的 tensor 以及原始文件路径
        return image_tensor, mask_tensor, image_path, mask_path

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    """ Device """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备进行评估: {device}")
    if torch.cuda.is_available():
        print(f"当前 GPU 名称: {torch.cuda.get_device_name(0)}")

    """ Hyperparameters (应与训练时一致) """
    image_size = 256  # 模型输入尺寸
    size = (image_size, image_size)  # (width, height)
    batch_size = 16  # 评估时 batch size 可以适当增大，因为它不涉及梯度计算

    # 数据集根路径 (包含 train 和 test 文件夹)
    # **请确认这个路径是正确的**
    dataset_root_path = r"C:/srp_OCT/segmentation_mask/segmentation_mask_dataset"  # 改为自己的路径

    # 定义需要评估的掩码类型及其对应的子文件夹名称
    # **请确认这些子文件夹名称与你的实际文件夹名称一致**
    mask_types_to_evaluate = {
        'anterior_chamber': 'qf_masks',
        'lens': 'jzt_masks',
        'cornea': 'jm_masks',
        'iris': 'hm_masks'
    }
    image_subfolder_name = 'images'  # 原始图像子文件夹名称

    # 训练好的模型检查点所在的目录
    # **请确认这个路径是正确的，它应该包含 checkpoint_mask_type.pth 文件**
    checkpoints_base_dir = "files_oct_segmentation"

    # 可视化结果保存的根目录
    output_results_base_dir = "evaluation_results_viz"  # 修改保存目录名称以区分
    create_dir(output_results_base_dir)

    # 定义每种掩码类型的可视化颜色 (BGR 格式, 0-255)
    # 尽量使用浅色，不要太鲜艳
    mask_colors = {
        'anterior_chamber': (200, 200, 100),  # 浅黄色
        'lens': (100, 200, 100),  # 浅绿色
        'cornea': (100, 100, 200),  # 浅蓝色
        'iris': (200, 100, 200)  # 浅紫色
        # 可以根据需要调整这些颜色
    }

    print("\n--- 开始评估模型 ---")

    # 数据集路径
    test_base_folder = os.path.join(dataset_root_path, 'test')

    # --- 循环评估每种掩码类型 ---
    for mask_type, mask_subfolder_name in mask_types_to_evaluate.items():
        print(f"\n--- 评估 {mask_type} 的分割模型 ---")

        # 检查点文件路径
        checkpoint_path = os.path.join(checkpoints_base_dir, f"checkpoint_{mask_type}.pth")

        # 检查检查点文件是否存在
        if not os.path.exists(checkpoint_path):
            print(f"错误: 未找到 {mask_type} 的模型检查点文件: {checkpoint_path}，跳过评估。")
            continue

        # 获取当前掩码类型的可视化颜色
        current_mask_color = mask_colors.get(mask_type, (128, 128, 128))  # 如果未定义颜色，使用灰色

        # 创建测试数据集实例
        test_dataset = OCTDataset(test_base_folder, image_subfolder_name, mask_subfolder_name, size)

        # 检查测试集是否为空
        if len(test_dataset) == 0:
            print(f"警告: {mask_type} 的测试集为空，无法进行评估。")
            continue

        # 创建 DataLoader
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,  # 评估时不需要打乱
            num_workers=2  # 根据你的机器性能调整 num_workers
        )

        # --- 创建可视化结果保存目录 ---
        mask_results_dir = os.path.join(output_results_base_dir, mask_type)
        joint_save_dir = os.path.join(mask_results_dir, "joint")
        mask_save_dir = os.path.join(mask_results_dir, "predicted_mask")  # 保存预测的二值掩码
        overlay_save_dir = os.path.join(mask_results_dir, "overlay")  # 保存叠加图

        create_dir(joint_save_dir)
        create_dir(mask_save_dir)
        create_dir(overlay_save_dir)
        print(f"可视化结果将保存在: {mask_results_dir}")

        """ Model """
        model = TResUnet()  # 实例化模型结构
        # 加载训练好的权重
        try:
            # 使用 map_location=device 确保权重加载到正确的设备
            # 添加 weights_only=True 参数
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            model = model.to(device)
            model.eval()  # 设置模型为评估模式 (关闭 dropout, batchnorm 等的训练行为)
            print(f"成功加载模型权重: {checkpoint_path}")
        except Exception as e:
            print(f"错误: 加载模型权重失败 {checkpoint_path}: {e}，跳过评估。")
            continue

        """ Evaluation Loop """
        # 根据 calculate_metrics 函数返回的指标数量初始化累加器
        # 假设 calculate_metrics 返回 Jaccard, F1, Recall, Precision, Accuracy, F2, Dice, IoU 共 8 个指标
        num_metrics = 6  # 增加指标数量
        total_metrics_score = [0.0] * num_metrics
        time_taken = []  # 用于计算FPS

        num_batches = len(test_loader)

        if num_batches == 0:
            print(f"警告: {mask_type} 的测试集 DataLoader 为空，无法进行评估。")
            continue

        with torch.no_grad():  # 在评估时不需要计算梯度，可以节省内存和计算
            for i, (x, y, img_paths, mask_paths) in tqdm(enumerate(test_loader), total=num_batches,
                                                         desc=f"Evaluating {mask_type}"):
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)

                # --- FPS calculation ---
                start_time = time.time()
                # 调用模型，不再请求 heatmap
                y_pred = model(x)  # 假设 model(x) 只返回预测 logits
                end_time = time.time() - start_time
                time_taken.append(end_time)

                # 将预测结果通过 sigmoid 转换为概率
                y_pred_prob = torch.sigmoid(y_pred)
                # 阈值化为二值掩码 (0或1)
                y_pred_binary = (y_pred_prob > 0.5).float()

                # --- 可视化保存 ---
                # 将 tensor 移到 CPU 并转换为 numpy
                x_np = x.cpu().numpy()  # (B, C, H, W)
                y_np = y.cpu().numpy()  # (B, 1, H, W)
                y_pred_binary_np = y_pred_binary.cpu().numpy()  # (B, 1, H, W)

                batch_size_actual = x.size(0)

                for j in range(batch_size_actual):
                    # 获取当前图像的文件名（不含扩展名）
                    img_filename = os.path.basename(img_paths[j])
                    name = os.path.splitext(img_filename)[0]

                    # 准备原始图像 (H, W, C), 0-255
                    # x_np 是 (B, C, H, W)，需要转置并缩放到 0-255
                    original_img_viz = np.transpose(x_np[j], (1, 2, 0)) * 255.0
                    original_img_viz = original_img_viz.astype(np.uint8)

                    # 准备地面真相掩码 (H, W, C), 0 或 255
                    # y_np 是 (B, 1, H, W)，需要移除通道维度，缩放到 0-255，并转为 3 通道
                    gt_mask_viz = np.squeeze(y_np[j], axis=0) * 255.0
                    gt_mask_viz = gt_mask_viz.astype(np.uint8)
                    gt_mask_viz = np.stack([gt_mask_viz] * 3, axis=-1)  # 转为 3 通道

                    # 准备预测掩码 (H, W, C), 0 或 255
                    # y_pred_binary_np 是 (B, 1, H, W)，需要移除通道维度，缩放到 0-255，并转为 3 通道
                    pred_mask_viz = np.squeeze(y_pred_binary_np[j], axis=0) * 255.0
                    pred_mask_viz = pred_mask_viz.astype(np.uint8)
                    pred_mask_viz_3channel = np.stack([pred_mask_viz] * 3, axis=-1)  # 转为 3 通道用于拼接

                    # --- 创建叠加图 (Overlay) ---
                    # 将预测的二值掩码 (0或255) 转换为布尔掩码 (True/False)
                    mask_bool = pred_mask_viz > 0
                    # 创建原始图像的副本用于叠加
                    overlay_img_viz = original_img_viz.copy()
                    # 将预测掩码为 True 的像素位置设置为指定的颜色
                    overlay_img_viz[mask_bool] = current_mask_color

                    # 创建分隔线
                    line = np.ones((size[1], 10, 3), dtype=np.uint8) * 255

                    # 拼接图像: 原始图像 | GT掩码 | 预测掩码 | 叠加图
                    # 所有数组现在都是 3 维 (H, W, C) 或 (H, 10, 3)
                    cat_images = np.concatenate([
                        original_img_viz, line,
                        gt_mask_viz, line,
                        pred_mask_viz_3channel, line,  # 使用 3 通道的预测掩码进行拼接
                        overlay_img_viz
                    ], axis=1)

                    # 保存拼接图像、预测掩码和叠加图
                    cv2.imwrite(os.path.join(joint_save_dir, f"{name}.jpg"), cat_images)
                    cv2.imwrite(os.path.join(mask_save_dir, f"{name}.png"), pred_mask_viz)  # 保存单通道预测掩码 (0或255)
                    cv2.imwrite(os.path.join(overlay_save_dir, f"{name}.jpg"), overlay_img_viz)

                # --- 计算指标 ---
                # calculate_metrics 期望单个图像的 tensor (1, H, W)
                batch_metrics = [0.0] * num_metrics  # 累加当前批次的指标

                for j in range(batch_size_actual):
                    # 提取批次中的单个图像和掩码
                    single_y_true = y[j]  # shape (1, H, W)
                    single_y_pred_binary = y_pred_binary[j]  # shape (1, H, W)

                    # 计算单个图像的指标
                    # calculate_metrics 应该期望 tensor 输入
                    score = calculate_metrics(single_y_true, single_y_pred_binary)

                    # **检查 calculate_metrics 返回的指标数量**
                    if len(score) != num_metrics:
                        print(
                            f"\n警告: calculate_metrics 函数返回的指标数量不匹配! 预期 {num_metrics}, 实际 {len(score)}. 请检查 utils.py 中的 calculate_metrics 函数.")
                        # 为了避免崩溃，这里只累加前 num_metrics 个指标 (如果返回的少于 num_metrics 会再次报错)
                        # 更好的做法是根据实际返回数量调整累加
                        score = score[:num_metrics]  # 截断或填充，这里选择截断

                    batch_metrics = list(map(add, batch_metrics, score))

                # 将批次指标平均后累加到总指标
                total_metrics_score = list(
                    map(add, total_metrics_score, [m / batch_size_actual for m in batch_metrics]))

        # 计算整个测试集上的平均指标
        # 注意：这里需要除以批次数量 num_batches，因为上面累加的是批次平均值
        avg_metrics = [m / num_batches for m in total_metrics_score]

        # 假设 calculate_metrics 返回顺序是 Jaccard, F1, Recall, Precision, Accuracy, F2,
        avg_iou = avg_metrics[0]
        avg_dice = avg_metrics[1]
        avg_recall = avg_metrics[2]
        avg_precision = avg_metrics[3]
        avg_acc = avg_metrics[4]
        avg_f2 = avg_metrics[5]

        # 计算平均FPS
        # time_taken 记录的是每个 batch 的推理时间，需要除以 batch size 得到每张图的平均时间
        mean_time_per_image = np.mean(time_taken) / batch_size if time_taken and batch_size > 0 else 0
        mean_fps = 1 / mean_time_per_image if mean_time_per_image > 0 else 0

        print(f"\n{mask_type} 测试集评估结果:")
        print(f"  IoU: {avg_iou:.4f}")
        print(f"  Dice: {avg_dice:.4f}")
        print(f"  Recall: {avg_recall:.4f}")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Accuracy: {avg_acc:.4f}")
        print(f"  F2 Score: {avg_f2:.4f}")
        print(f"  平均推理时间 (每张图像): {mean_time_per_image:.4f} 秒")
        print(f"  平均 FPS: {mean_fps:.2f}")

    print("\n所有掩码类型的模型评估完成。")
