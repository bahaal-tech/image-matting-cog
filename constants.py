# CV2
import os
from dotenv import load_dotenv

load_dotenv()
HAR_CASCADES_PATH = "haarcascade_frontalface_default.xml"
FACE_SCALE_FACTOR = 1.1
MIN_NEIGHBOURS = 5
MIN_SIZE = (30, 30)

LOWER_SKIN_BOUNDARIES = [0, 133, 77]
UPPER_SKIN_BOUNDARIES = [255, 173, 127]

# EMBEDDING
MATRIX_TRANSFORM = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
SIMILARITY_DIMENSION = 0

# VIT_MATTE
VIT_MATTE_MODEL_NAME = "vitmatte-s"
VIT_MATTE_MODEL_CHECKPOINT = "/checkpoints/ViTMatte_S_Com.pth"

# HSV
THRESHOLD = 60

# DIRECTORY
DIRECTORY_TO_SAVE_VIT_MATTE = "./vit-matte-results"
MODEL_DIR = "./checkpoints/ViTMatte_S_Com.pth"
DIRECTORY_TO_SAVE_MODIFIED_MATTE = "./modified_matte"
DIRECTORY_TO_SAVE_EDGE_LESS_MATTE = "./edgeless_matte"
DIRECTORY_TO_SAVE_IMAGE_OVERLAY = "./image_overlay"

# EMBEDDING MODEL
EMBEDDING_MODEL_NAME = "imagenet"
POLLING_EMBEDDING = "avg"
EMBEDDING_THRESHOLD = 0.70

# SENTRY
SENTRY_DSN = os.getenv("SENTRY_DSN")

# WANDB
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
