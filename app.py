# app.py - Flask приложение для детекции Deepfake
import os
import io
import base64
import tempfile
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import traceback
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from collections import defaultdict

# Импорты из model.py
from model import (
    XceptionSE, SEBlock, create_xception_model, crop_face_from_frame, 
    sample_frames_fixed_interval, VAL_TEST_TRANSFORM, set_seed, 
    validate_video_level, get_target_layer, get_device, FRAMES_PER_VIDEO
)

# Попытка импорта Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    print("WARNING: pytorch-grad-cam not available, heatmaps will be disabled")
    GRADCAM_AVAILABLE = False

# Конфигурация
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Глобальные переменные для модели
MODEL = None
DEVICE = None
MODEL_NAME = "Xception"
MAX_HEATMAPS = 10

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model(model_path, device):
    """Загрузка обученной модели Xception"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = create_xception_model(pretrained=False, device=device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def convert_to_python_types(obj):
    """Рекурсивно конвертирует numpy типы в Python типы для JSON сериализации"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_to_python_types(item) for item in obj]
    return obj

def generate_gradcam_batch(model, frames_data, device, target_layer):
    """Batch генерация Grad-CAM для multiple frames"""
    if not GRADCAM_AVAILABLE:
        return []
    
    try:
        cam = GradCAM(model=model, target_layers=target_layer)
        results = []
        
        for frame_info in frames_data:
            input_tensor = frame_info['input_tensor']
            
            # Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            
            # Накладываем на оригинальное изображение
            orig_frame = frame_info['orig_frame'].astype(np.float32) / 255.0
            heatmap = show_cam_on_image(orig_frame, grayscale_cam, use_rgb=True)
            
            results.append({
                'heatmap': heatmap,
                'confidence': float(frame_info['confidence']),
                'pred_class': int(frame_info['pred_class']),
                'frame_idx': int(frame_info['frame_idx'])
            })
        
        # Очистка ресурсов
        if hasattr(cam, 'activations_and_grads'):
            cam.activations_and_grads.release()
        
        return results
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        traceback.print_exc()
        return []

def pil_to_base64(img, format='PNG'):
    """Конвертация PIL Image в base64 строку"""
    try:
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"
    except Exception as e:
        print(f"Error converting to base64: {e}")
        return None

def numpy_to_base64(arr, format='PNG'):
    """Конвертация numpy array в base64 строку"""
    try:
        # Убедимся, что значения в правильном диапазоне
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        return pil_to_base64(img, format)
    except Exception as e:
        print(f"Error converting numpy to base64: {e}")
        traceback.print_exc()
        return None

def detect_deepfake(video_path, model, device, max_heatmaps=10):
    """
    Основная функция детекции deepfake
    """
    print(f"Starting analysis of: {video_path}")
    
    # 1. Извлечение кадров с лицами
    frame_indices = sample_frames_fixed_interval(video_path, FRAMES_PER_VIDEO)
    print(f"Will analyze frames at indices: {frame_indices[:5]}... (total: {len(frame_indices)})")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {
            'success': False,
            'error': 'Cannot open video file',
            'video_info': {}
        }
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info: {total_frames} frames, {fps} fps, {duration:.2f}s duration")
    
    frames_data = []
    faces_detected = 0
    frames_processed = 0
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to read frame {idx}")
            continue
        
        frames_processed += 1
        
        # Детекция лица
        try:
            face_pil = crop_face_from_frame(frame)
        except Exception as e:
            print(f"Error in face detection for frame {idx}: {e}")
            continue
        
        if face_pil is None:
            continue
        
        faces_detected += 1
        
        # Предобработка
        try:
            orig_frame = np.array(face_pil)
            input_tensor = VAL_TEST_TRANSFORM(face_pil).unsqueeze(0).to(device)
            
            # Предсказание
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.softmax(output, dim=1)[0]
                fake_prob = float(prob[1].item())  # Конвертируем в Python float
                pred_class = 1 if fake_prob >= 0.5 else 0
                confidence = fake_prob if pred_class == 1 else 1 - fake_prob
            
            frames_data.append({
                'frame_idx': int(idx),
                'timestamp': float(idx / fps if fps > 0 else 0),
                'orig_frame': orig_frame,
                'input_tensor': input_tensor,
                'fake_prob': fake_prob,
                'confidence': float(confidence),
                'pred_class': int(pred_class)
            })
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            traceback.print_exc()
            continue
    
    cap.release()
    
    print(f"Processed {frames_processed} frames, found {faces_detected} faces, valid: {len(frames_data)}")
    
    if not frames_data:
        return {
            'success': False,
            'error': 'No faces detected in video',
            'video_info': {
                'total_frames': int(total_frames),
                'fps': float(fps),
                'duration': float(duration),
                'frames_processed': int(frames_processed)
            }
        }
    
    # 2. Video-level агрегация (mean_prob)
    mean_fake_prob = float(np.mean([f['fake_prob'] for f in frames_data]))
    is_fake = bool(mean_fake_prob >= 0.5)  # Явно конвертируем в Python bool
    final_confidence = float(mean_fake_prob if is_fake else 1 - mean_fake_prob)
    
    print(f"Aggregation: mean_fake_prob={mean_fake_prob:.4f}, is_fake={is_fake}, confidence={final_confidence:.4f}")
    
    # 3. Выбор топ-k кадров для визуализации
    frames_data.sort(key=lambda x: abs(x['fake_prob'] - 0.5), reverse=True)
    selected_frames = frames_data[:max_heatmaps]
    
    print(f"Selected top {len(selected_frames)} frames for visualization")
    
    # 4. Генерация Grad-CAM
    heatmap_results = []
    if GRADCAM_AVAILABLE:
        try:
            target_layer = get_target_layer(model, MODEL_NAME)
            heatmap_results = generate_gradcam_batch(model, selected_frames, device, target_layer)
            print(f"Generated {len(heatmap_results)} heatmaps")
        except Exception as e:
            print(f"Error generating heatmaps: {e}")
            traceback.print_exc()
    
    # 5. Формируем результат
    explanations = []
    for i, frame_info in enumerate(selected_frames):
        try:
            exp_data = {
                'index': i + 1,
                'frame_number': int(frame_info['frame_idx']),
                'timestamp': round(frame_info['timestamp'], 2),
                'fake_probability': round(frame_info['fake_prob'] * 100, 2),
                'prediction': 'FAKE' if frame_info['pred_class'] == 1 else 'REAL',
                'confidence': round(frame_info['confidence'] * 100, 2),
            }
            
            # Конвертируем изображения в base64
            orig_b64 = numpy_to_base64(frame_info['orig_frame'])
            exp_data['original_image'] = orig_b64 if orig_b64 else ''
            
            # Heatmap (если есть)
            if i < len(heatmap_results):
                heat_b64 = numpy_to_base64(heatmap_results[i]['heatmap'])
                exp_data['heatmap_image'] = heat_b64 if heat_b64 else ''
            else:
                exp_data['heatmap_image'] = ''
            
            explanations.append(exp_data)
        except Exception as e:
            print(f"Error creating explanation {i}: {e}")
            traceback.print_exc()
            continue
    
    # Статистика по всем кадрам
    fake_frames = int(sum(1 for f in frames_data if f['pred_class'] == 1))
    
    result = {
        'success': True,
        'result': {
            'is_fake': is_fake,
            'verdict': 'FAKE' if is_fake else 'REAL',
            'confidence': round(final_confidence * 100, 2),
            'fake_probability': round(mean_fake_prob * 100, 2),
            'aggregation_method': 'mean_prob'
        },
        'statistics': {
            'total_frames_analyzed': int(len(frames_data)),
            'faces_detected': int(faces_detected),
            'frames_predicted_fake': fake_frames,
            'frames_predicted_real': int(len(frames_data) - fake_frames),
            'video_duration_seconds': round(float(duration), 2),
            'video_fps': round(float(fps), 2),
            'video_total_frames': int(total_frames)
        },
        'explanations': explanations
    }
    
    # Конвертируем все numpy типы в Python типы
    result = convert_to_python_types(result)
    
    print(f"Analysis complete: {result['result']['verdict']} with {result['result']['confidence']}% confidence")
    return result

# ============ ROUTES ============

@app.route('/')
def index():
    """Главная страница с интерфейсом загрузки"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint для анализа видео"""
    print(f"Received analyze request, files: {list(request.files.keys())}")
    
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False, 
            'error': f'Invalid file format. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
        }), 400
    
    # Сохраняем временный файл
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(temp_path)
        print(f"Saved uploaded file to: {temp_path}, size: {os.path.getsize(temp_path)} bytes")
        
        # Параметры
        max_heatmaps = request.form.get('max_heatmaps', 10, type=int)
        max_heatmaps = min(max(max_heatmaps, 1), 20)  # ограничение 1-20
        
        print(f"Parameters: max_heatmaps={max_heatmaps}")
        
        # Анализируем
        result = detect_deepfake(temp_path, MODEL, DEVICE, max_heatmaps=max_heatmaps)
        
        # Удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Cleaned up temp file: {temp_path}")
        
        return jsonify(result)
        
    except Exception as e:
        # Логируем полную ошибку
        print(f"ERROR in analyze: {e}")
        traceback.print_exc()
        
        # Удаляем временный файл при ошибке
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL is not None,
        'device': str(DEVICE),
        'model_name': MODEL_NAME,
        'gradcam_available': GRADCAM_AVAILABLE
    })

# ============ INITIALIZATION ============

def init_app(model_path):
    """Инициализация приложения с загрузкой модели"""
    global MODEL, DEVICE
    
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    MODEL = load_model(model_path, DEVICE)
    print("Model loaded successfully!")
    
    return app

if __name__ == '__main__':
    # Путь к вашей обученной модели
    MODEL_PATH = "/home/vladix35/Diploma/output/Xception/faces/training/Xception.pth"
    
    init_app(MODEL_PATH)
    app.run(host='0.0.0.0', port=5000, debug=True)