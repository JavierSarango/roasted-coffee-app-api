
# Coffee Roast Classifier API
# ===========================


# API desarrollada con FastAPI para la clasificación del grado de tostado del café a partir de imágenes.
# El sistema integra:
# - Un modelo UNet para segmentación de granos de café.
# - Un modelo CNN para clasificación del tostado.
# - Validaciones geométricas y métricas de incertidumbre para filtrar entradas inválidas.


# La API recibe una imagen, segmenta los granos, valida el contenido, clasifica el tostado y devuelve
# la predicción junto con métricas y una visualización de la segmentación en formato Base64.


import io
import cv2
import base64
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

# ==========================================
# CONFIGURACIÓN DE LOGGING
# ==========================================
"""
Configura el sistema de logging para registrar eventos informativos y errores
relacionados con el ciclo de vida de la API y el procesamiento de inferencias.
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURACIÓN GLOBAL
# ==========================================
"""
Define parámetros globales del sistema:
- Dimensiones de entrada de imagen
- Parámetros de validación geométrica
- Umbrales de confianza e incertidumbre
- Configuración de clases
"""

IMG_SIZE = (128, 128)
CHANNELS = 3
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MIN_OBJECTS_FOR_LOOSE_CHECK = 3


SINGLE_OBJ_MAX_AREA_RATIO = 0.95
SINGLE_OBJ_MIN_SOLIDITY = 0.88
SINGLE_OBJ_MIN_AR = 0.5
SINGLE_OBJ_MAX_AR = 2.0


MULTI_OBJ_MIN_SOLIDITY = 0.60
MULTI_OBJ_MIN_AREA_RATIO = 0.05


MC_DROPOUT_ITERATIONS = 15
MAX_STD_DEV = 0.20
CONF_THRESHOLD = 0.60


CLASS_NAMES = {
0: 'Dark',
1: 'Green',
2: 'Light',
3: 'Medium',
4: 'Overbaking'
}


BASE_DIR = Path(__file__).resolve().parent

UNET_WEIGHTS = f"{BASE_DIR}/unet_segmentation_model_best.pth"
CNN_WEIGHTS = f"{BASE_DIR}/cnn_segmented_cnn_(optimizado)_best.pth"

# ==========================================
# DEFINICIÓN DE ARQUITECTURAS
# ==========================================
"""
Bloque convolucional básico utilizado en la arquitectura UNet.
Combina convoluciones 2D y funciones de activación ReLU.
"""
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    
"""
Arquitectura UNet simplificada para segmentación binaria de granos de café.
Produce una máscara que identifica las regiones relevantes de la imagen.
"""
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.e1 = conv_block(in_channels, 16); self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e2 = conv_block(16, 32)
        self.bridge = conv_block(32, 64)
        self.up_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d1 = conv_block(32 + 32, 32)
        self.up_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.d2 = conv_block(16 + 16, 16)
        self.out = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.e1(x); p1 = self.maxpool(c1)
        c2 = self.e2(p1); p2 = self.maxpool(c2)
        c3 = self.bridge(p2)
        u4 = self.up_conv1(c3); u4 = torch.cat([u4, c2], dim=1); c4 = self.d1(u4)
        u5 = self.up_conv2(c4); u5 = torch.cat([u5, c1], dim=1); c5 = self.d2(u5)
        return self.out(c5)

"""
Arquitectura CNN para clasificación del grado de tostado del café.
Incluye Dropout para permitir estimación de incertidumbre mediante MC Dropout.
"""

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==========================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ==========================================
"""
Inicializa la aplicación FastAPI, configura CORS y define el esquema
estándar de respuesta para el endpoint de predicción.
"""

app = FastAPI(title="Coffee Roast Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    clase_predicha: str
    confianza: float
    segmentacion_base64: str  
    mensaje_error: str = ""
    metricas_validacion: dict = {}

"""
Evento de arranque de la aplicación.
Carga los modelos de segmentación y clasificación en memoria
para evitar recargas en cada solicitud.
"""
@app.on_event("startup")
def load_models():
    logger.info(f"Iniciando API en dispositivo: {DEVICE}")
    app.state.unet = UNet(in_channels=CHANNELS, out_channels=1).to(DEVICE)
    app.state.cnn = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
    
    try:
        u_ckpt = torch.load(UNET_WEIGHTS, map_location=DEVICE)
        app.state.unet.load_state_dict(u_ckpt.get('model_state_dict', u_ckpt))
        app.state.unet.eval()
        
        c_ckpt = torch.load(CNN_WEIGHTS, map_location=DEVICE)
        app.state.cnn.load_state_dict(c_ckpt.get('model_state_dict', c_ckpt))
        app.state.cnn.eval()
        logger.info("Modelos cargados correctamente.")
    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")

# ==========================================
# FUNCIONES UTILITARIAS DE PREPROCESAMIENTO
# ==========================================
"""
Normaliza un tensor de imagen utilizando una transformación lineal
centrada en cero, adecuada para la inferencia de los modelos entrenados.
"""

def normalize(tensor):
    return (tensor - 0.5) / 0.5

"""
Convierte una imagen PIL a un tensor de PyTorch, realizando:
- Conversión de espacio de color RGB a Lab
- Normalización de valores
- Reordenamiento de dimensiones
"""

def process_pil_to_tensor(image_pil: Image.Image) -> torch.Tensor:
    img_np = np.array(image_pil)
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2Lab)
    img_float = img_lab.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img_float, (2, 0, 1))).float()
    return tensor.unsqueeze(0).to(DEVICE)
    
"""
Genera una visualización superpuesta entre la imagen original y la máscara
segmentada, codificada en Base64 para su envío al frontend.
"""
def create_overlay_image(original_pil: Image.Image, mask_tensor: torch.Tensor) -> str:
    img_np = np.array(original_pil) 
    mask_np = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8) 
    
    if np.sum(mask_np) > 0:
        color_mask = np.zeros_like(img_np)
        color_mask[mask_np > 128] = [0, 255, 0] 
        overlay = cv2.addWeighted(img_np, 0.7, color_mask, 0.3, 0)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    else:
        overlay = img_np

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", overlay_bgr)
    if not success: return ""
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

# ==========================================
# VALIDACIÓN DE CONTENIDO DE LA IMAGEN
# ==========================================
"""
Analiza la máscara segmentada para validar que el contenido corresponde
realmente a granos de café y no a objetos ajenos o ruido visual.


Se consideran métricas geométricas como:
- Número de objetos
- Proporción de área ocupada
- Convexidad y relación de aspecto
"""

def analyze_coffee_content(mask_tensor_cpu):

    mask_np = (mask_tensor_cpu.numpy() * 255).astype(np.uint8)
    total_area = mask_np.shape[0] * mask_np.shape[1]
    
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, "No se detectó ningún objeto", {}

    valid_contours = [c for c in contours if cv2.contourArea(c) > 30]
    num_objects = len(valid_contours)
    
    mask_pixel_count = np.sum(mask_np > 0)
    mask_ratio = mask_pixel_count / total_area

    metrics = {
        "num_objects": num_objects,
        "mask_ratio": mask_ratio
    }

    
    if mask_ratio > SINGLE_OBJ_MAX_AREA_RATIO:
        return False, "Posible fondo o textura (Área excesiva)", metrics

   
    if num_objects < MIN_OBJECTS_FOR_LOOSE_CHECK:
        c = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: return False, "Error geométrico", metrics
        
        solidity = float(area) / hull_area
        x,y,w,h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        
        metrics["solidity"] = solidity
        metrics["aspect_ratio"] = aspect_ratio
        
        if solidity < SINGLE_OBJ_MIN_SOLIDITY:
            return False, f"Objeto único irregular. No parece un grano de café.", metrics
        
        if aspect_ratio < SINGLE_OBJ_MIN_AR or aspect_ratio > SINGLE_OBJ_MAX_AR:
             return False, f"Objeto único deforme", metrics
    else:

        if mask_ratio < MULTI_OBJ_MIN_AREA_RATIO:
             return False, "Muy poco contenido detectado", metrics

    return True, "OK", metrics

# ==========================================
# PREDICCIÓN CON INCERTIDUMBRE
# ==========================================
"""
Realiza inferencia utilizando MC Dropout para estimar la incertidumbre
asociada a la predicción del modelo de clasificación.
"""

def predict_with_uncertainty(model, input_tensor, num_samples=10):
    model.train() # Habilitar Dropout
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.cpu().numpy())
    
    model.eval()
    predictions = np.vstack(predictions)
    
    mean_probs = np.mean(predictions, axis=0)
    std_dev = np.std(predictions, axis=0)
    
    best_class_idx = np.argmax(mean_probs)
    mean_confidence = mean_probs[best_class_idx]
    uncertainty_score = std_dev[best_class_idx]
    
    return best_class_idx, mean_confidence, uncertainty_score

# ==========================================
# ENDPOINT PRINCIPAL DE PREDICCIÓN
# ==========================================
"""
Endpoint principal de la API.


Recibe una imagen, ejecuta la segmentación, valida el contenido,
clasifica el grado de tostado y retorna el resultado junto con
métricas y visualización de la segmentación.
"""

@app.post("/predict", response_model=PredictionResponse)
async def predict_roast(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        image_pil = image_pil.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        
        input_raw = process_pil_to_tensor(image_pil)

        # A. Segmentación
        app.state.unet.eval()
        with torch.no_grad():
            unet_input = normalize(input_raw)
            mask_logits = app.state.unet(unet_input)
            mask = (torch.sigmoid(mask_logits) > 0.5).float()
            
        overlay_b64 = create_overlay_image(image_pil, mask)

        # B. Análisis Inteligente de Contenido
        is_valid_content, reason, metrics = analyze_coffee_content(mask.squeeze().cpu())
        
        if not is_valid_content:
            logger.info(f"Rechazo de contenido: {reason} | Métricas: {metrics}")
            return PredictionResponse(
                clase_predicha="No Coffee",
                confianza=0.0,
                segmentacion_base64=overlay_b64,
                mensaje_error=reason,
                metricas_validacion=metrics
            )
        
        segmented_raw = input_raw * mask
        cnn_input = normalize(segmented_raw)
        
        pred_idx, confidence, uncertainty = predict_with_uncertainty(
            app.state.cnn, cnn_input, num_samples=MC_DROPOUT_ITERATIONS
        )
        
        metrics["uncertainty"] = float(uncertainty)

        if uncertainty > MAX_STD_DEV:
             logger.info(f"Rechazo por Incertidumbre: {uncertainty:.3f}")
             return PredictionResponse(
                clase_predicha="Unknown",
                confianza=float(confidence),
                segmentacion_base64=overlay_b64,
                mensaje_error="Objeto ambiguo (Alta incertidumbre del modelo)",
                metricas_validacion=metrics
            )

        if confidence < CONF_THRESHOLD:
             return PredictionResponse(
                clase_predicha="Unknown",
                confianza=float(confidence),
                segmentacion_base64=overlay_b64,
                mensaje_error=f"Confianza insuficiente ({confidence:.1%})",
                metricas_validacion=metrics
            )

        class_name = CLASS_NAMES.get(pred_idx, "Error")
        
        return PredictionResponse(
            clase_predicha=class_name,
            confianza=float(confidence),
            segmentacion_base64=overlay_b64,
            mensaje_error="",
            metricas_validacion=metrics
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

# ==========================================
# ENDPOINT DE ESTADO
# ==========================================
"""
Endpoint básico de verificación de estado de la API.
"""
@app.get("/")
def root():
    return {"status": "online"}