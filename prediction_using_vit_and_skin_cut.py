import torch
import torch.nn as nn
from constants import VIT_MATTE_MODEL_NAME, THRESHOLD, DIRECTORY_TO_SAVE_VIT_MATTE, \
    DIRECTORY_TO_SAVE_MODIFIED_MATTE, EMBEDDING_THRESHOLD, MODEL_DIR
from utils import model_initializer, alpha_matte_inference_from_vision_transformer, \
    selective_search_and_remove_skin_tone, calculate_embeddings_diff_between_two_images
import torchvision.models as models


class SkinSegmentVitMatte:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vit_matte_model = model_initializer(model=VIT_MATTE_MODEL_NAME, checkpoint=MODEL_DIR,
                                                 device=self.device)
        resnet = models.resnet50(pretrained=True)
        self.embedding_model = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding_model.eval()

    def generate_modified_matted_results(self, input_image, trimap_image):
        cutout_image_from_vit_matting = alpha_matte_inference_from_vision_transformer(self.vit_matte_model, input_image,
                                                                                      trimap_image,
                                                                                      DIRECTORY_TO_SAVE_VIT_MATTE)
        if not cutout_image_from_vit_matting["success"]:
            return {"success": False, "error": f"Vit matting failed due to: {cutout_image_from_vit_matting}"}
        modified_matte = selective_search_and_remove_skin_tone(input_image,
                                                               cutout_image_from_vit_matting["vit_matte_output"],
                                                               THRESHOLD, DIRECTORY_TO_SAVE_MODIFIED_MATTE)
        if not modified_matte["success"]:
            return {"success": False, "error": f"matting modifications failed due to:{modified_matte['error']}"}

        distance_between_modified_and_vit_matte = calculate_embeddings_diff_between_two_images(
            modified_matte["output"], cutout_image_from_vit_matting["vit_matte_output"], self.embedding_model)
        if not distance_between_modified_and_vit_matte["success"]:
            return {"success": True, "vit_matte_path": cutout_image_from_vit_matting["vit_matte_output"],
                    "modified_matte_path": modified_matte["output"], "embedding_check_label": False, "error_reason":
                    distance_between_modified_and_vit_matte["error"]}
        if distance_between_modified_and_vit_matte["cosine_distance"] > EMBEDDING_THRESHOLD:
            return {"success": True, "vit_matte_path": cutout_image_from_vit_matting["vit_matte_output"],
                    "modified_matte_path": "", "embedding_check_label": True, "error_reason": ""}
        else:
            return {"success": True, "vit_matte_path": cutout_image_from_vit_matting["vit_matte_output"],
                    "modified_matte_path": modified_matte["output"], "embedding_check_label": True, "error_reason": ""}
