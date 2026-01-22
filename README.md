# ☕ Roasted Coffee App - API Backend

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

Este repositorio contiene la **API Backend** para la aplicación *Roasted Coffee App*. Su función principal es recibir imágenes de granos de café desde el cliente web, procesarlas utilizando modelos de Deep Learning y devolver el nivel de tostado o clasificación Agtron.

## 🧠 Modelos Integrados

El servicio carga y utiliza los siguientes modelos entrenados en PyTorch:

*   **UNet Segmentation (`unet_segmentation_model_best.pth`)**: Se encarga de segmentar la imagen para aislar los granos de café del fondo.
*   **CNN Classifier (`cnn_segmented_cnn_(optimizado)_best.pth`)**: Analiza los granos segmentados para determinar su clasificación o características.
*   **Modelo Final (`modelo_cafe_pytorch_final.pth`)**: Modelo de clasificación binaria para distinguir entre imágenes que contienen granos de café y las que no (filtrado inicial)..

## 🚀 Características del API

*   📡 **Endpoints REST**: Comunicación con el frontend.
*   🖼️ **Procesamiento de Imágenes**: Preprocesamiento, redimensionamiento y normalización automática.
*   ⚡ **Inferencia en Tiempo Real**: Carga de modelos optimizada para respuestas rápidas.
*   🐳 **Containerización**: Listo para despliegue (si usas Docker/Render/Railway).

## 🛠️ Instalación y Ejecución Local

Sigue estos pasos para levantar el servidor en tu máquina:

### Pasos 
1. **Clonar el repositorio**
```bash
git clone https://github.com/JavierSarango/roasted-coffee-app-api.git
cd roasted-coffee-app-api
```
2. **Crear entorno virtual (Recomendado)**
```bash
## python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```
3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

5. **Ejecutar servidor**
```bash
# Opción A (FastAPI/Uvicorn recomendado):
uvicorn main:app --reload

# Opción B (Si tienes un bloque if __name__ == "__main__"):
python main.py
```

El servidor debería estar corriendo en `http://localhost:8000` (o el puerto configurado).

## 🔗 Endpoints Principales


| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/` | Health Check - Verifica que la API está activa. |
| `POST` | `/predict` | Recibe una imagen (multipart/form-data) y devuelve la predicción. |

## 📂 Estructura del Proyecto

```text
roasted-coffee-app-api/
├── .github/workflows/   # CI/CD pipelines
├── main.py              # Punto de entrada de la aplicación (API)
├── requirements.txt     # Librerías necesarias
├── *.pth                # Archivos de modelos entrenados (Pesos)
└── .gitignore           # Archivos ignorados por Git
```

## 🤝 Relación con otros repositorios

Este proyecto es parte del ecosistema *Roasted Coffee*:

1.  **Frontend**: [roasted-coffee-app](https://github.com/JavierSarango/roasted-coffee-app) (Interfaz de usuario).
2.  **Research**: [roast-coffee-agtron](https://github.com/JavierSarango/roast-coffee-agtron) (Entrenamiento y notebooks).
3.  **Backend (Este repo)**: API de inferencia.
