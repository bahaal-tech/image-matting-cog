import torch
import os
import cv2
import logging
import torch.nn as nn
from constants import VIT_MATTE_MODEL_NAME, THRESHOLD, DIRECTORY_TO_SAVE_VIT_MATTE, \
    DIRECTORY_TO_SAVE_MODIFIED_MATTE, EMBEDDING_THRESHOLD, MODEL_DIR, DIRECTORY_TO_SAVE_EDGE_LESS_MATTE
from logger import get_system_usage
from utils import model_initializer, alpha_matte_inference_from_vision_transformer, \
    selective_search_and_remove_skin_tone, calculate_embeddings_diff_between_two_images, \
    extra_edge_removal_from_matte_output, convert_greyscale_image_to_transparent
import torchvision.models as models

if torch.cuda.is_available():
    scaler = torch.cuda.amp.GradScaler()


class SkinSegmentVitMatte:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vit_matte_model = model_initializer(model=VIT_MATTE_MODEL_NAME, checkpoint=MODEL_DIR,
                                                 device=self.device)
        resnet = models.resnet50(pretrained=True)
        self.embedding_model = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding_model.eval()

    def generate_modified_matted_results(self, input_image, trimap_image):
        with torch.cuda.amp.autocast():
            cutout_image_from_vit_matting = alpha_matte_inference_from_vision_transformer(self.vit_matte_model,
                                                                                          input_image,
                                                                                          trimap_image,
                                                                                          DIRECTORY_TO_SAVE_VIT_MATTE)
            compute_for_vit_matte = get_system_usage()
            logging.info(f"VIT Matte compute: {compute_for_vit_matte}")
        torch.cuda.empty_cache()
        if not cutout_image_from_vit_matting["success"]:
            return {"success": False, "error": f"Vit matting failed due to: {cutout_image_from_vit_matting}"}
        with torch.cuda.amp.autocast():
            modified_matte = selective_search_and_remove_skin_tone(input_image,
                                                                   cutout_image_from_vit_matting["vit_matte_output"],
                                                                   THRESHOLD, DIRECTORY_TO_SAVE_MODIFIED_MATTE)
            modified_matte_compute = get_system_usage()
            logging.info(f"Modified Matte compute: {modified_matte_compute}")
        torch.cuda.empty_cache()
        logging.info(f"cutout image from vit matting is {cutout_image_from_vit_matting}")

        if not modified_matte["success"]:
            modified_matte_path_need_to_be_passed = cutout_image_from_vit_matting["vit_matte_output"]
        else:
            modified_matte_path_need_to_be_passed = modified_matte["output"]
        modified_matte_image = cv2.imread(modified_matte_path_need_to_be_passed)
        kernel_for_modified_matte = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        final_image = cv2.morphologyEx(modified_matte_image, cv2.MORPH_OPEN, kernel_for_modified_matte, iterations=3)
        dir_final = os.path.join(DIRECTORY_TO_SAVE_MODIFIED_MATTE, "final_matte_image.png")
        cv2.imwrite(dir_final, final_image)
        grey_scale_final_path = os.path.join(DIRECTORY_TO_SAVE_MODIFIED_MATTE, "edge_less_final_matte.png")
        convert_greyscale_image_to_transparent(dir_final, grey_scale_final_path)
        distance_between_modified_and_vit_matte = calculate_embeddings_diff_between_two_images(
            modified_matte_path_need_to_be_passed, grey_scale_final_path, self.embedding_model)
        if not distance_between_modified_and_vit_matte["success"]:
            return {"success": True, "vit_matte_path": cutout_image_from_vit_matting["vit_matte_output"],
                    "edge_less_no_mask": grey_scale_final_path,
                    "modified_matte_path": grey_scale_final_path, "embedding_check_label": False, "error_reason":
                        distance_between_modified_and_vit_matte["error"], "distance": ""}
        if distance_between_modified_and_vit_matte["cosine_distance"]["similarity"] > EMBEDDING_THRESHOLD:
            return {"success": True, "vit_matte_path": cutout_image_from_vit_matting["vit_matte_output"],
                    "edge_less_no_mask": grey_scale_final_path,
                    "modified_matte_path": grey_scale_final_path, "embedding_check_label": True, "error_reason": "",
                    "distance": distance_between_modified_and_vit_matte["cosine_distance"]["similarity"]}
        else:
            return {"success": True, "vit_matte_path": cutout_image_from_vit_matting["vit_matte_output"],
                    "edge_less_no_mask": grey_scale_final_path,
                    "modified_matte_path": grey_scale_final_path, "embedding_check_label": True, "error_reason": "",
                    "distance": distance_between_modified_and_vit_matte["cosine_distance"]["similarity"]
                    }
