import os
import random
from PIL import Image
import cv2
import torch
import sentry_sdk
import wandb
import numpy as np
from PIL import Image
from os.path import join as opj
from torchvision.transforms import functional as F
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from constants import HAR_CASCADES_PATH, FACE_SCALE_FACTOR, MIN_NEIGHBOURS, MIN_SIZE, LOWER_SKIN_BOUNDARIES, \
    UPPER_SKIN_BOUNDARIES, MATRIX_TRANSFORM, NORMALIZE_MEAN, NORMALIZE_STD, SIMILARITY_DIMENSION, SENTRY_DSN

sentry_sdk.init(
    dsn=SENTRY_DSN,
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)


def generate_inference_from_one_image(model, input_image, dir_to_save=None):
    """
        Infer the alpha matte of one image.
        Input:
            model: the trained model
            image: the input image
            trimap: the input trimap
    """
    output = model(input_image)['phas'].flatten(0, 2)
    output = F.to_pil_image(output)
    output.save(opj(dir_to_save))
    return None


def model_initializer(model, checkpoint, device):
    """
    Initialize the model.
    Input:
        config: the config file of the model
        checkpoint: the checkpoint of the model
    """
    assert model in ['vitmatte-s', 'vitmatte-b']
    if model == 'vitmatte-s':
        config = 'ViTMatte/configs/common/model.py'
        cfg = LazyConfig.load(config)
        model = instantiate(cfg.model)
        model.to(device)
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    elif model == 'vitmatte-b':
        config = 'ViTMatte/configs/common/model.py'
        cfg = LazyConfig.load(config)
        cfg.model.backbone.embed_dim = 768
        cfg.model.backbone.num_heads = 12
        cfg.model.decoder.in_chans = 768
        model = instantiate(cfg.model)
        model.to(device)
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    return model


def generate_model_input(input_image, image_trimap):
    """
    Get the data of one image.
    Input:
        image_dir: the directory of the image
        trimap_dir: the directory of the trimap
    """
    image = Image.open(input_image).convert('RGB')
    image_tensor = F.to_tensor(image).unsqueeze(0)
    trimap = Image.open(image_trimap).convert('L')
    trimap_tensor = F.to_tensor(trimap).unsqueeze(0)

    return {
        'image': image_tensor,
        'trimap': trimap_tensor
    }


def calculate_foreground(input_image, alpha_matte, output_path):
    """
    Calculate the foreground of the image.
    Input:
        image_dir: the directory of the image
        alpha_dir: the directory of the alpha matte
        save_path: the path to save the resulting foreground image
    Output:
        None
    """
    image = Image.open(input_image).convert('RGB')
    alpha = Image.open(alpha_matte).convert('L')
    alpha = F.to_tensor(alpha).unsqueeze(0)
    image = F.to_tensor(image).unsqueeze(0)
    foreground = image * alpha + (1 - alpha)
    foreground = foreground.squeeze(0).permute(1, 2, 0).numpy()
    image_cutout = (foreground * 255).astype(np.uint8)
    cv2.imwrite(output_path, image_cutout)


def alpha_matte_inference_from_vision_transformer(model, input_image, trimap_image, directory_to_save):
    try:
        input_to_vit_model = generate_model_input(input_image, trimap_image)
        if not os.path.exists(directory_to_save):
            os.mkdir(directory_to_save)
        save_dir_for_matte = os.path.join(directory_to_save, 'matte.png')
        generate_inference_from_one_image(model, input_to_vit_model, save_dir_for_matte)
        dir_for_alpha_output = os.path.join(directory_to_save, 'alpha.png')
        dir_for_cutout_output = os.path.join(directory_to_save, 'cutout.png')
        convert_greyscale_image_to_transparent(save_dir_for_matte, dir_for_alpha_output)
        calculate_foreground(input_image, dir_for_alpha_output, dir_for_cutout_output)
        return {"success": True, "vit_matte_output": dir_for_alpha_output, "cutout_output": dir_for_cutout_output}
    except Exception as e:
        error_message = f"Vit Matte model failed due : {e}"
        raise_sentry_error(e)
        return {"success": False, "error": error_message}


def detect_face_and_hsv_from_images(image):
    input_image = cv2.imread(str(image))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAR_CASCADES_PATH)
    gray_scale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    try:
        faces = face_cascade.detectMultiScale(gray_scale_image, scaleFactor=FACE_SCALE_FACTOR,
                                              minNeighbors=MIN_NEIGHBOURS, minSize=MIN_SIZE)
        if len(faces) == 0:
            return {"success": False, "error": f"Number of face detected: {len(faces)}"}
        for (x, y, w, h) in faces:
            region_of_interest = input_image[y:y + h, x:x + w]
            roi_ycrcb = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2YCrCb)
            lower_skin = np.array(LOWER_SKIN_BOUNDARIES, dtype=np.uint8)
            upper_skin = np.array(UPPER_SKIN_BOUNDARIES, dtype=np.uint8)
            mask_skin = cv2.inRange(roi_ycrcb, lower_skin, upper_skin)
            skin_color = cv2.mean(region_of_interest, mask=mask_skin)[:3]
            skin_color = np.uint8([[skin_color]])
            skin_color_hsv = cv2.cvtColor(skin_color, cv2.COLOR_BGR2HSV)
            h, s, v = skin_color_hsv[0][0]
            return {"success": True, "h_value": h, "s_value": s, "v_value": v}
    except Exception as e:
        error_message = f"Face detection or hsv extraction failed due to :{e}"
        raise_sentry_error(e)
        return {"success": False, "error": f"Face detection or hsv extraction failed due to :{e}"}


def selective_search_and_remove_skin_tone(input_image, matte_image, threshold_for_hsv, directory_to_save):
    hsv_values = detect_face_and_hsv_from_images(input_image)
    if not hsv_values["success"]:
        return {"success": False, "error": f"Either no face detected or might be internal error, "
                                           f"kindly refer:{hsv_values['error']}"}
    h_value = hsv_values["h_value"]
    s_value = hsv_values["s_value"]
    v_value = hsv_values["v_value"]

    image_matte = cv2.imread(str(matte_image))
    hsv_of_matte = cv2.cvtColor(image_matte, cv2.COLOR_BGR2HSV)
    target_hsv = np.array([h_value, s_value, v_value])

    lower_hsv = np.array([target_hsv[0] - threshold_for_hsv, target_hsv[1] - threshold_for_hsv,
                          target_hsv[2] - threshold_for_hsv])
    upper_hsv = np.array([target_hsv[0] + threshold_for_hsv, target_hsv[1] + threshold_for_hsv,
                          target_hsv[2] + threshold_for_hsv])

    mask = cv2.inRange(hsv_of_matte, lower_hsv, upper_hsv)
    inverse_mask = cv2.bitwise_not(mask)  # Invert the mask
    result = np.copy(image_matte)
    result[inverse_mask == 0] = [255, 255, 255]  # Keep the non-masked area

    if not os.path.exists(directory_to_save):
        os.mkdir(directory_to_save)
    output_dir_for_modified_matte = os.path.join(directory_to_save, 'modified_cutout.png')
    cv2.imwrite(output_dir_for_modified_matte, result)
    return {"success": True, "output": output_dir_for_modified_matte}


def get_image_embeddings(model, image_path):
    import torchvision.transforms as transforms
    preprocess_input_image = transforms.Compose([
        transforms.Resize(MATRIX_TRANSFORM),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD), ])
    try:
        image = Image.open(image_path)
        img = preprocess_input_image(image).unsqueeze(0)
        with torch.no_grad():
            embedding = model(img)
        return {"success": True, "embeddings": embedding.squeeze().numpy()}
    except Exception as e:
        return {"success": False, "error": f"Image embedding creation failed due to {e}"}


def calculate_cosine_distance(modified_matting_image, vit_matted_image):
    try:
        embedding_new_matte = torch.from_numpy(modified_matting_image)
        embedding_vit_matte = torch.from_numpy(vit_matted_image)
        similarity = torch.nn.functional.cosine_similarity(embedding_new_matte, embedding_vit_matte,
                                                           dim=SIMILARITY_DIMENSION)
        return {"success": True, "similarity": similarity.item()}
    except Exception as e:
        return {"success": False, "error": f"Calculation of cosine similarity failed due to{e}"}


def calculate_embeddings_diff_between_two_images(modified_matting_image, vit_matted_image, model):
    modified_image_embedding = get_image_embeddings(model, modified_matting_image)
    vit_matte_image_embedding = get_image_embeddings(model, vit_matted_image)
    if modified_image_embedding["success"] and vit_matte_image_embedding["success"]:
        cosine_distance = calculate_cosine_distance(modified_image_embedding["embeddings"],
                                                    vit_matte_image_embedding["embeddings"])
        if not cosine_distance["success"]:
            return {"success": False, "error": f"Similarity check failed due to :{cosine_distance['error']}"}
        return {"success": True, "cosine_distance": cosine_distance}
    else:
        return {"success": False, "error": f"Similarity check failed due to {modified_image_embedding} "
                                           f"{vit_matte_image_embedding}"}


def convert_greyscale_image_to_transparent(input_image_path, output_path):
    """
    Converts a greyscale image to a transparent one. Makes
    absolute black pixels transparent, absolute white opaque
    and grey ones to intermediate alpha values.
    Input:
        input_image_path: Path on disk for input greyscale image
        output_path: The path where transparent image needs to be written
    """
    input_image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_RGB2RGBA)
    (image_height, image_width, _) = input_image.shape
    alpha_image = np.zeros([image_height, image_width, 4])
    alpha_channel = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    color_channel = np.where(alpha_channel > 0, 255, 0)

    alpha_image[:, :, 0] = color_channel
    alpha_image[:, :, 1] = color_channel
    alpha_image[:, :, 2] = color_channel
    alpha_image[:, :, 3] = alpha_channel
    cv2.imwrite(output_path, alpha_image)


def extra_edge_removal_from_matte_output(matte_image, output_path):
    from rembg import new_session, remove
    try:
        output_path_for_non_mask_edge_less_image = os.path.join(output_path, "edge_less_no_mask.png")
        output_path_for_mask_edge_less_image = os.path.join(output_path, "edge_less_mask.png")
        image = cv2.imread(matte_image, cv2.IMREAD_COLOR)
        output = remove(
            image,
            alpha_matting=True,
            alpha_matting_foreground_threshold=270,
            alpha_matting_background_threshold=-10,
            alpha_matting_erode_size=-10,
            epsilon=1e-5,
            session=new_session("u2net"),
            only_mask=True
        )
        cv2.imwrite(output_path_for_mask_edge_less_image, output)
        output_no_mask = remove(
            image,
            alpha_matting=True,
            alpha_matting_foreground_threshold=270,
            alpha_matting_background_threshold=-10,
            alpha_matting_erode_size=-10,
            epsilon=1e-5,
            session=new_session("u2net"),
            only_mask=False
        )
        cv2.imwrite(output_path_for_non_mask_edge_less_image, output_no_mask)
        return {"success": True, "mask_edge_less_path": output_path_for_mask_edge_less_image, "non_mask_edge_less_path":
                output_path_for_non_mask_edge_less_image}
    except Exception as e:
        error_message = f"Edge removal failed due to :{e}"
        raise_sentry_error(e)
        return {"success": False, "error": f"Edge removal failed due to :{e}"}

def overlay_final_mask_in_black_background(image_path, mask_path, output_image_path):
    """
    This function is designed to create cutout of garment from the original image using
    the alpha-matte mask and finally overlay it in black background. Idea is same like
    adding white/grey background in frontend, here we are using black so that we can see
    black lines.
    :param image_path: Actual Image --> loaded as PIL array (RGBA)
    :param mask_path: Alpha Matte (final one) --> loaded as PIL array (RGBA)
    :param output_image_path: Saving path
    :return: cutout image in black background --> Converted from PIL array with RGBA
    """
    try:
        actual_image = Image.open(image_path).convert("RGBA")
        mask_image = Image.open(mask_path).convert("RGBA")
        mask_alpha = mask_image.split()[-1]
        background_color = (0, 0, 0, 255)
        segmented_image = Image.new("RGBA", actual_image.size, background_color)
        actual_image_array = np.array(actual_image)
        mask_alpha_array = np.array(mask_alpha)
        for channel in range(3):
            actual_image_array[:, :, channel] = actual_image_array[:, :, channel] * (mask_alpha_array / 255.0)
        masked_image = Image.fromarray(actual_image_array, "RGBA")
        segmented_image.paste(masked_image, (0, 0), mask_alpha)
        segmented_image.save(output_image_path)
        return {"success": True, "overlay_path": output_image_path}
    except Exception as e:
        return {"success": False, "error": f"Overlay failed due to {e}"}


def raise_sentry_error(error_message):
    sentry_sdk.capture_exception(error_message)

def initialize_wandb(product_id):
    experiment_count = random.random()
    wandb.init(project=f"{product_id}_matte_{experiment_count}", ntags=["Matting_Experiment"])

def log_results_to_wandb(output_image_name, path_of_image):
    wandb.log({f"{output_image_name}": wandb.Image(path_of_image)})
