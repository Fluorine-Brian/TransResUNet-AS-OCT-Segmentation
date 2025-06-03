
import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from metrics import precision, recall, F2, dice_score, jac_score
from sklearn.metrics import accuracy_score, confusion_matrix
import sys # 导入 sys 用于打印到标准输出
import io # 导入 io 用于检查文件对象类型

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True ## For reproducibility
    torch.backends.cudnn.benchmark = False ## For reproducibility

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Shuffle the dataset. """
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_handle_or_path, text):
    """
    打印文本到控制台，并写入到文件句柄或指定路径的文件。

    Args:
        file_handle_or_path: 一个已经打开的文件句柄 (如由 open() 返回)，或者一个文件路径 (str)。
        text (str): 要打印和写入的文本。
    """
    # 打印到控制台
    print(text)

    # 写入到文件
    if isinstance(file_handle_or_path, (str, bytes, os.PathLike)):
        # 如果是路径，则打开文件并写入
        try:
            with open(file_handle_or_path, "a") as file: # 使用追加模式 "a"
                file.write(text + "\n")
        except Exception as e:
            print(f"错误: 无法写入文件 {file_handle_or_path}: {e}", file=sys.stderr) # 打印错误到标准错误
    elif isinstance(file_handle_or_path, io.TextIOWrapper):
        # 如果是文件句柄，则直接写入
        try:
            file_handle_or_path.write(text + "\n")
            file_handle_or_path.flush() # 立即将缓冲区内容写入文件
        except Exception as e:
             print(f"错误: 无法写入文件句柄: {e}", file=sys.stderr)
    else:
        print(f"警告: print_and_save 接收到未知类型的第一个参数: {type(file_handle_or_path)}", file=sys.stderr)

def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_iou = jac_score(y_true, y_pred)
    score_dice = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)


    return [score_iou, score_dice, score_recall, score_precision, score_acc, score_fbeta]
