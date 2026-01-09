import io
import cv2
import base64
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import convnext_tiny # Importación necesaria
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURACIÓN GLOBAL
# ==========================================
IMG_SIZE_BINARY = (224, 224) 
IMG_SIZE = (128, 128)
CHANNELS = 3
NUM_CLASSES_ROAST = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Rutas de Pesos ---
BASE_DIR = Path(__file__).resolve().parent
UNET_WEIGHTS = f"{BASE_DIR}/unet_segmentation_model_best.pth"
CNN_ROAST_WEIGHTS = f"{BASE_DIR}/cnn_segmented_cnn_(optimizado)_best.pth"
BINARY_WEIGHTS = f"{BASE_DIR}/modelo_cafe_pytorch_final.pth" 

CLASS_NAMES = {0: 'Dark', 1: 'Green', 2: 'Light', 3: 'Medium', 4: 'Overbaking'}

# ==========================================
# DEFINICIÓN DE ARQUITECTURAS
# ==========================================

# --- 1. UNet (Segmentación) ---
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    
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

# --- 2. Roast Classifier (CNN Original) ---
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

# --- 3. Binary Classifier (ConvNeXt Tiny) ---
def build_binary_model():

    model = convnext_tiny(weights=None)

    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        1
    )
    return model

# ==========================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ==========================================

app = FastAPI(title="Coffee Roast Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.on_event("startup")
def load_models():
    logger.info(f"Iniciando API en dispositivo: {DEVICE}")
    
    # 1. Instanciar Modelos
    app.state.unet = UNet(in_channels=CHANNELS, out_channels=1).to(DEVICE)
    app.state.roast_cnn = SimpleCNN(num_classes=NUM_CLASSES_ROAST).to(DEVICE)
    app.state.binary_cnn = build_binary_model().to(DEVICE) 
    
    try:
        # 2. Cargar Pesos
        # UNet
        u_ckpt = torch.load(UNET_WEIGHTS, map_location=DEVICE)
        app.state.unet.load_state_dict(u_ckpt.get('model_state_dict', u_ckpt))
        app.state.unet.eval()
        
        # Roast CNN
        c_ckpt = torch.load(CNN_ROAST_WEIGHTS, map_location=DEVICE)
        app.state.roast_cnn.load_state_dict(c_ckpt.get('model_state_dict', c_ckpt))
        app.state.roast_cnn.eval()

        # Binary ConvNeXt
        if Path(BINARY_WEIGHTS).exists():
            b_ckpt = torch.load(BINARY_WEIGHTS, map_location=DEVICE)
            if isinstance(b_ckpt, dict) and 'model_state_dict' in b_ckpt:
                b_ckpt = b_ckpt['model_state_dict']
            app.state.binary_cnn.load_state_dict(b_ckpt)
            app.state.binary_cnn.eval()
            logger.info("Todos los modelos (UNet, RoastCNN, ConvNeXt) cargados correctamente.")
        else:
            logger.warning(f"ADVERTENCIA: No se encontró {BINARY_WEIGHTS}")

    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")

# ==========================================
# UTILIDADES
# ==========================================
def normalize(tensor):
    return (tensor - 0.5) / 0.5

def process_pil_to_tensor(image_pil: Image.Image) -> torch.Tensor:
    img_np = np.array(image_pil)
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2Lab)
    img_float = img_lab.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img_float, (2, 0, 1))).float()
    return tensor.unsqueeze(0).to(DEVICE)
    
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
# PREDICCIÓN CON INCERTIDUMBRE (Solo Tostado)
# ==========================================
def predict_with_uncertainty(model, input_tensor, num_samples=10):
    model.train() 
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
# ENDPOINT PRINCIPAL
# ==========================================

@app.post("/predict", response_model=PredictionResponse)
async def predict_roast(file: UploadFile = File(...)):
    try:
        contents = await file.read()
    # ------------------------------------------------------------------
        # RAMA A: PREPARACIÓN PARA FILTRO BINARIO (224x224)
        # ------------------------------------------------------------------
        # Redimensionamos específicamente para ConvNeXt
        pil_binary = original_pil.resize(IMG_SIZE_BINARY, Image.Resampling.BILINEAR)
        input_binary = process_pil_to_tensor(pil_binary)
        input_binary_norm = normalize(input_binary)

        # ------------------------------------------------------------------
        # RAMA B: PREPARACIÓN PARA TOSTADO (128x128)
        # ------------------------------------------------------------------
        # Redimensionamos para UNet y RoastCNN
        pil_roast = original_pil.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        input_roast = process_pil_to_tensor(pil_roast)
        input_roast_norm = normalize(input_roast)

       # ------------------------------------------------------------------
        # FASE 1: FILTRO BINARIO (Usando entrada 224)
        # ------------------------------------------------------------------
        with torch.no_grad():
            binary_logit = app.state.binary_cnn(input_binary_norm)
            coffee_prob = torch.sigmoid(binary_logit).item()
            
        if coffee_prob < 0.5:
            dummy_overlay = create_overlay_image(pil_roast, torch.zeros((1, 1, 128, 128)))
            
            logger.info(f"Rechazo Binario: Probabilidad {coffee_prob:.2f}")
            
            user_msg = "No se logró identificar granos de café en la imagen. Por favor, asegúrate de que los granos estén centrados y bien iluminados."
            if coffee_prob > 0.2:
                user_msg = "Parece café, pero la imagen no es clara. Intenta acercar la cámara y enfocar mejor los granos."

            return PredictionResponse(
                clase_predicha="No Coffee",
                confianza=1.0 - coffee_prob,
                segmentacion_base64=dummy_overlay,
                mensaje_error=user_msg,
                metricas_validacion={"binary_score": coffee_prob}
            )

        # ------------------------------------------------------------------
        # FASE 2: SEGMENTACIÓN (Usando entrada 128)
        # ------------------------------------------------------------------
        app.state.unet.eval()
        with torch.no_grad():
            mask_logits = app.state.unet(input_roast_norm)
            mask = (torch.sigmoid(mask_logits) > 0.5).float()
            
        overlay_b64 = create_overlay_image(pil_roast, mask)

        # ------------------------------------------------------------------
        # FASE 3: CLASIFICACIÓN DE TOSTADO (Usando entrada 128)
        # ------------------------------------------------------------------
        segmented_raw = input_roast * mask  
        cnn_input = normalize(segmented_raw)
        
        pred_idx, confidence, uncertainty = predict_with_uncertainty(
            app.state.roast_cnn, cnn_input, num_samples=15
        )
        
        class_name = CLASS_NAMES.get(pred_idx, "Error")
        
        return PredictionResponse(
            clase_predicha=class_name,
            confianza=float(confidence),
            segmentacion_base64=overlay_b64,
            mensaje_error="",
            metricas_validacion={"uncertainty": float(uncertainty), "coffee_prob": coffee_prob}
        )

    except Exception as e:
        logger.error(f"Error crítico: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/")
def root():
    return {"status": "online"}