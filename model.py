# model.py - Модуль с моделью и утилитами для детекции Deepfake
import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, roc_curve, auc,
    average_precision_score, precision_recall_curve
)
import timm

# ============ КОНФИГУРАЦИЯ ============

RANDOM_SEED = 42
FRAMES_PER_VIDEO = 30
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

TRAIN_RATIO = 0.75
VAL_RATIO = 0.125
TEST_RATIO = 0.125

# Глобальные устройства и детекторы
_device = None
_mtcnn = None

def get_device():
    """Получение устройства (GPU/CPU)"""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device

def get_mtcnn():
    """Получение/инициализация MTCNN детектора лиц"""
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=False, device=get_device())
    return _mtcnn

def set_seed(seed=RANDOM_SEED):
    """Установка seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============ ТРАНСФОРМАЦИИ ============

_aug_resize_256 = transforms.Resize((256, 256))
_to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

FACE_TRAIN_TRANSFORM = transforms.Compose([
    _aug_resize_256,
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    _to_tensor_norm,
])

FRAME_TRAIN_TRANSFORM = transforms.Compose([
    _aug_resize_256,
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    _to_tensor_norm,
])

VAL_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    _to_tensor_norm,
])

# ============ ДЕТЕКЦИЯ ЛИЦ ============

def _detect_face_bbox(frame: np.ndarray):
    """
    Детекция лица на кадре с помощью MTCNN
    Returns: (x1, y1, x2, y2) или None
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = get_mtcnn().detect(rgb)
    
    if boxes is not None and len(boxes) > 0:
        return tuple(boxes[0])  # Первое обнаруженное лицо
    return None

def crop_face_from_frame(frame: np.ndarray, fallback_size=(224, 224), margin=0.2):
    """
    Вырезание лица из кадра с отступом (margin)
    Returns: PIL.Image или None
    """
    bbox = _detect_face_bbox(frame)
    if bbox is None:
        return None
    
    x1, y1, x2, y2 = bbox
    h, w, _ = frame.shape
    
    # Добавляем отступ
    dw, dh = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(x1 - dw, 0)), int(max(y1 - dh, 0))
    x2, y2 = int(min(x2 + dw, w)), int(min(y2 + dh, h))
    
    # Вырезаем и ресайзим
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, fallback_size)
    
    return Image.fromarray(face_resized)

def sample_frames_fixed_interval(video_path: str, frames_per_video: int = FRAMES_PER_VIDEO):
    """
    Равномерная выборка кадров из видео
    Returns: список индексов кадров
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if total <= 0:
        return []
    
    step = max(total // frames_per_video, 1)
    indices = [i * step for i in range(frames_per_video) if i * step < total]
    return indices

# ============ DATASET ============

class DeepFakeFrameDataset(Dataset):
    """
    Dataset для кадров видео
    Returns: (tensor_image, label, video_path)
    """
    
    def __init__(self, frame_info_list, transform=None, face_crop=True):
        """
        frame_info_list: список (video_path, frame_num, label)
        transform: torchvision transforms
        face_crop: если True — вырезать лицо, иначе использовать весь кадр
        """
        self.frame_info_list = frame_info_list
        self.transform = transform
        self.face_crop = face_crop
    
    def __len__(self):
        return len(self.frame_info_list)
    
    def __getitem__(self, idx):
        video_path, frame_num, label = self.frame_info_list[idx]
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_pil = None
        attempt = 0
        max_attempts = 5
        
        while attempt < max_attempts and frame_pil is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if self.face_crop:
                frame_pil = crop_face_from_frame(frame)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
            
            # Если не удалось получить кадр, пробуем случайный другой
            if frame_pil is None:
                frame_num = random.randint(0, max(total_frames - 1, 0))
                attempt += 1
        
        cap.release()
        
        # Fallback: если всё не удалось, возвращаем заглушку
        if frame_pil is None:
            frame_pil = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            frame_pil = self.transform(frame_pil)
        
        return frame_pil, torch.tensor(label, dtype=torch.long), video_path

# ============ МОДЕЛЬ ============

class SEBlock(nn.Module):
    """Squeeze-and-Excitation блок для attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class XceptionSE(nn.Module):
    """Xception с SE-блоком для детекции Deepfake"""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.se = SEBlock(channels=2048, reduction=16)
        self.fc = nn.Linear(2048, 2)  # 2 класса: real/fake
    
    def forward(self, x):
        # Извлекаем признаки
        f = self.base.forward_features(x)
        # Применяем attention
        f = self.se(f)
        # Global pooling
        p = self.base.global_pool(f)
        p = torch.flatten(p, 1)
        # Классификация
        return self.fc(p)

def create_xception_model(pretrained=True, device=None):
    """
    Создание модели XceptionSE
    """
    if device is None:
        device = get_device()
    
    # Загружаем базовую модель
    backbone = timm.create_model('xception', pretrained=pretrained)
    
    # Замораживаем все слои по умолчанию
    for p in backbone.parameters():
        p.requires_grad = False
    
    # Размораживаем последние блоки для fine-tuning
    unfreeze_modules = ['block6', 'block7', 'block8', 'conv4', 'bn4']
    for name, module in backbone.named_children():
        if name in unfreeze_modules:
            for p in module.parameters():
                p.requires_grad = True
    
    model = XceptionSE(backbone).to(device)
    return model

def get_target_layer(model, model_name="Xception"):
    """
    Получение target layer для Grad-CAM
    """
    if model_name == "Xception":
        # Для Xception используем последний conv слой
        return [model.base.conv4]
    elif model_name == "ResNet50":
        return [model.layer4[-1]]
    else:
        # Fallback: предпоследний слой
        return [list(model.children())[-2]]

# ============ ВАЛИДАЦИЯ И ТЕСТИРОВАНИЕ ============

def validate_video_level(model, val_loader, device):
    """
    Валидация на уровне видео с разными методами агрегации
    """
    model.eval()
    video_dict = defaultdict(lambda: {"labels": [], "probs": [], "preds": []})
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    
    with torch.no_grad():
        for images, labels, video_paths in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            for i, vp in enumerate(video_paths):
                video_dict[vp]["labels"].append(labels[i].item())
                video_dict[vp]["probs"].append(probs[i].cpu().numpy())
                video_dict[vp]["preds"].append(preds[i].item())
    
    val_loss = running_loss / total if total > 0 else 0.0
    
    # Методы агрегации
    methods = {
        'mean_prob': lambda x: np.mean([p[1] for p in x]),
        'max_prob': lambda x: np.max([p[1] for p in x]),
        'min_prob': lambda x: np.min([p[1] for p in x]),
        'median_prob': lambda x: np.median([p[1] for p in x]),
        'trimmed_mean_prob': lambda x: np.mean(sorted([p[1] for p in x])[1:-1]) if len(x) > 2 else np.mean([p[1] for p in x]),
        'vote_percentage': lambda x: np.mean(x)
    }
    
    metrics_all = {}
    best_f1 = -1.0
    best_acc = 0.0
    best_method = None
    
    for name, agg_fn in methods.items():
        y_true, y_pred, scores = [], [], []
        
        for info in video_dict.values():
            if not info["labels"]:
                continue
            
            true_label = info["labels"][0]
            
            # Агрегация
            if name == 'vote_percentage':
                score = agg_fn(info["preds"])
            else:
                score = agg_fn(info["probs"])
            
            pred_label = 1 if score >= 0.5 else 0
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            scores.append(score)
        
        if not y_true:
            continue
        
        # Метрики
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # ROC и AUC
        y_bin = np.array(y_true)
        fpr, tpr, _ = roc_curve(y_bin, scores)
        roc_auc = auc(fpr, tpr)
        
        # EER (Equal Error Rate)
        eer = fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]
        
        # Average Precision
        ap = average_precision_score(y_bin, scores)
        
        metrics_all[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'mcc': mcc,
            'auc': roc_auc,
            'ap': ap,
            'eer': eer
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_method = name
    
    return val_loss, best_acc, best_f1, best_method, metrics_all

# ============ ОБУЧЕНИЕ ============

def train_model(model, train_loader, val_loader, device, output_folder, 
                num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
    """
    Полный цикл обучения модели
    """
    os.makedirs(output_folder, exist_ok=True)
    model_path = os.path.join(output_folder, "Xception.pth")
    
    # Проверяем, есть ли уже обученная модель
    if os.path.exists(model_path):
        tqdm.write("Xception already trained; loading.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Читаем лучший метод агрегации
        best_method = "mean_prob"  # default
        metrics_file = os.path.join(output_folder, "Xception_validation_metrics.txt")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                for line in f:
                    if line.startswith("Best:"):
                        best_method = line.split(":")[1].strip()
                        break
        
        return model, best_method
    
    # Настройка обучения
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, 
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )
    scaler = GradScaler()
    
    # Early stopping
    early_stop_patience = 5
    no_improve = 0
    best_val_loss = float('inf')
    best_method_overall = "mean_prob"
    
    # История обучения
    train_losses, train_accs = [], []
    val_losses, val_accs, val_f1s = [], [], []
    
    for epoch in range(num_epochs):
        # ===== TRAIN =====
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix(
                loss=f"{running_loss/total:.4f}", 
                acc=f"{correct/total:.4f}"
            )
        
        train_losses.append(running_loss / total)
        train_accs.append(correct / total)
        
        # ===== VALIDATION =====
        val_loss, val_acc, val_f1, best_method, metrics_all = validate_video_level(
            model, val_loader, device
        )
        
        scheduler.step(val_loss)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train loss {train_losses[-1]:.4f}, acc {train_accs[-1]:.4f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.4f}, f1 {val_f1:.4f} ({best_method})"
        )
        
        # Сохраняем метрики
        metrics_file = os.path.join(output_folder, "Xception_validation_metrics.txt")
        with open(metrics_file, "a") as f:
            f.write(f"\nEpoch {epoch+1}\n")
            for m_name, met in metrics_all.items():
                f.write(
                    f"{m_name} -> Acc: {met['accuracy']:.4f}, "
                    f"Prec: {met['precision']:.4f}, Rec: {met['recall']:.4f}, "
                    f"F1: {met['f1']:.4f}, MCC: {met['mcc']:.4f}, "
                    f"AUC: {met['auc']:.4f}, AP: {met['ap']:.4f}, EER: {met['eer']:.4f}\n"
                )
            f.write(f"Best: {best_method}\n")
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_method_overall = best_method
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            tqdm.write(f"Saved new best model (loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                tqdm.write("Early stopping triggered.")
                break
    
    # ===== ПЛОТЫ =====
    try:
        import matplotlib.pyplot as plt
        
        # Loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'training_loss.png'))
        plt.close()
        
        # Accuracy curves
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
        plt.plot(range(1, len(val_accs)+1), val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'training_accuracy.png'))
        plt.close()
        
        # F1 curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(val_f1s)+1), val_f1s, label='Val F1', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'validation_f1.png'))
        plt.close()
        
    except ImportError:
        tqdm.write("matplotlib not available, skipping plots")
    
    return model, best_method_overall

# ============ ПОДГОТОВКА ДАННЫХ ============

def _gather_frames(video_list, require_face=True):
    """
    Сбор информации о кадрах из списка видео
    """
    info = []
    
    for vid_path, label in tqdm(video_list, desc="Processing videos", unit="video"):
        frame_indices = sample_frames_fixed_interval(vid_path)
        
        for idx in frame_indices:
            if not require_face:
                info.append((vid_path, idx, label))
                continue
            
            # Проверяем, есть ли лицо на кадре
            cap = cv2.VideoCapture(vid_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret and crop_face_from_frame(frame) is not None:
                info.append((vid_path, idx, label))
    
    return info

def prepare_data_faces(real_dataset_path: str, fake_dataset_path: str):
    """
    Подготовка данных с вырезанными лицами
    """
    set_seed()
    
    real_videos = sorted([
        os.path.join(real_dataset_path, f) 
        for f in os.listdir(real_dataset_path) 
        if f.endswith('.mp4')
    ])
    fake_videos = sorted([
        os.path.join(fake_dataset_path, f) 
        for f in os.listdir(fake_dataset_path) 
        if f.endswith('.mp4')
    ])
    
    # Создаём список (video_path, label)
    vids = [(v, 0) for v in real_videos] + [(v, 1) for v in fake_videos]
    random.shuffle(vids)
    
    # Разбиваем
    n_total = len(vids)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)
    
    train_v = vids[:n_train]
    val_v = vids[n_train:n_train + n_val]
    test_v = vids[n_train + n_val:]
    
    tqdm.write(f"[faces] split: {len(train_v)} train / {len(val_v)} val / {len(test_v)} test")
    
    # Собираем кадры
    train_info = _gather_frames(train_v, require_face=True)
    val_info = _gather_frames(val_v, require_face=True)
    test_info = _gather_frames(test_v, require_face=True)
    
    tqdm.write(f"[faces] frames: train={len(train_info)} val={len(val_info)} test={len(test_info)}")
    
    # Создаём датасеты
    train_ds = DeepFakeFrameDataset(train_info, FACE_TRAIN_TRANSFORM, face_crop=True)
    val_ds = DeepFakeFrameDataset(val_info, VAL_TEST_TRANSFORM, face_crop=True)
    test_ds = DeepFakeFrameDataset(test_info, VAL_TEST_TRANSFORM, face_crop=True)
    
    return train_ds, val_ds, test_ds

def prepare_data_full_frame(real_dataset_path: str, fake_dataset_path: str):
    """
    Подготовка данных с полными кадрами (без детекции лиц)
    """
    set_seed()
    
    real_videos = sorted([
        os.path.join(real_dataset_path, f) 
        for f in os.listdir(real_dataset_path) 
        if f.endswith('.mp4')
    ])
    fake_videos = sorted([
        os.path.join(fake_dataset_path, f) 
        for f in os.listdir(fake_dataset_path) 
        if f.endswith('.mp4')
    ])
    
    vids = [(v, 0) for v in real_videos] + [(v, 1) for v in fake_videos]
    random.shuffle(vids)
    
    n_total = len(vids)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)
    
    train_v = vids[:n_train]
    val_v = vids[n_train:n_train + n_val]
    test_v = vids[n_train + n_val:]
    
    tqdm.write(f"[frames] split: {len(train_v)} train / {len(val_v)} val / {len(test_v)} test")
    
    train_info = _gather_frames(train_v, require_face=False)
    val_info = _gather_frames(val_v, require_face=False)
    test_info = _gather_frames(test_v, require_face=False)
    
    tqdm.write(f"[frames] frames: train={len(train_info)} val={len(val_info)} test={len(test_info)}")
    
    train_ds = DeepFakeFrameDataset(train_info, FRAME_TRAIN_TRANSFORM, face_crop=False)
    val_ds = DeepFakeFrameDataset(val_info, VAL_TEST_TRANSFORM, face_crop=False)
    test_ds = DeepFakeFrameDataset(test_info, VAL_TEST_TRANSFORM, face_crop=False)
    
    return train_ds, val_ds, test_ds

# ============ MAIN PIPELINE ============

def Xception_main(train_ds, val_ds, test_ds, device, output_folder, model_path=None):
    """
    Основной pipeline для Xception
    """
    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    
    # Создаём модель
    model = create_xception_model(pretrained=True, device=device)
    
    # Если указан путь — загружаем веса
    if model_path and os.path.exists(model_path):
        tqdm.write(f"Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        best_method = "mean_prob"
    else:
        # Обучаем
        model, best_method = train_model(
            model, train_loader, val_loader, device, output_folder
        )
    
    return test_loader, model, best_method

# ============ ЭКСПОРТ ДЛЯ WEB ============

__all__ = [
    # Модель
    'XceptionSE', 'SEBlock', 'create_xception_model',
    # Детекция лиц
    'crop_face_from_frame', '_detect_face_bbox', 'sample_frames_fixed_interval',
    # Трансформы
    'VAL_TEST_TRANSFORM', 'FACE_TRAIN_TRANSFORM', 'FRAME_TRAIN_TRANSFORM',
    # Датасет
    'DeepFakeFrameDataset',
    # Утилиты
    'set_seed', 'get_device', 'get_mtcnn', 'get_target_layer',
    'validate_video_level', 'train_model',
    # Подготовка данных
    'prepare_data_faces', 'prepare_data_full_frame', 'Xception_main',
    # Константы
    'FRAMES_PER_VIDEO', 'BATCH_SIZE', 'NUM_EPOCHS', 'LEARNING_RATE'
]