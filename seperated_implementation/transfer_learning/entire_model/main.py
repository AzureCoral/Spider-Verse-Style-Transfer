import tensorflow as tf
from helpers import tensor_to_image, load_img, powerset
from model import StyleTransfer
import time
from typing import Dict, List, Tuple

def load_style_content_imgs(style_links: Dict[str, str], content_link: Dict[str, str]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """
    Loads the style and content images.

    Parameters:
    style_links (Dict[str, str]): A dictionary where keys are names of the style images and values are their URLs.
    content_link (str): The URL of the content image.

    Returns:
    Tuple[tf.Tensor, List[tf.Tensor]]: The content image tensor and the style image tensors.
    """
    content_title, content_link = list(content_link.items())[0]
    content_path = tf.keras.utils.get_file(content_title, content_link)
    style_paths = [tf.keras.utils.get_file(key, style_links[key]) for key in style_links]
    content_image = load_img(content_path)
    style_images = [load_img(style_path) for style_path in style_paths]
    return content_title, content_image, style_images

def style_transfer(style_links: Dict[str, str], content_link: Dict[str, str], visuals: bool = False) -> None:
    """
    Performs style transfer from the style images to the content image.

    Parameters:
    style_links (Dict[str, str]): A dictionary where keys are names of the style images and values are their URLs.
    content_link (str): The URL of the content image.
    visuals (bool, optional): If True, visualizes the training process. Defaults to False.
    """
    # Load the content and style images:
    print("Loading images", end='\r', flush=True)
    content_title, content_image, style_images = load_style_content_imgs(style_links, content_link)
    print("Images loaded.")

    # Define content and style representations
    content_layers: List[str] = [
        'block5_conv2'
    ] 

    style_layers: List[str] = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1', 
        'block4_conv1', 
        'block5_conv1'
    ]

    # Initialize the model:
    print("Initializing model", end='\r', flush=True)
    model = StyleTransfer(style_layers, content_layers, style_images, content_image, 1e-2, 1e4)
    print("Model initialized.")

    # Train the model:
    print("\nTraining model:")
    img = model.train(visuals=visuals)    
    print("\nModel trained.")

    # Save the output image:
    output_file_path = f"basic/output/{content_title}_{int(time.time())}.jpg"
    print("Saving the output image", end='\r', flush=True)
    with open(output_file_path,'wb') as f:
        tensor_to_image(img).save(f, "JPEG")
    print(f"Output image saved at {output_file_path}")

def hyperparameter_search(
    style_links: Dict[str, str], 
    content_link: str, 
    style_weights: List[float], 
    content_weights: List[float], 
    total_variance_weights: List[float], 
    visuals: bool = False) -> None:
    """
    Performs style transfer with different hyperparameters from the style images to the content image.

    Parameters:
    style_links (Dict[str, str]): A dictionary where keys are names of the style images and values are their URLs.
    content_link (str): The URL of the content image.
    style_weights (List[float]): List of style weights to try.
    content_weights (List[float]): List of content weights to try.
    total_variance_weights (List[float]): List of total variance weights to try.
    visuals (bool, optional): If True, visualizes the training process. Defaults to False.
    """
    # Load the content and style images:
    print("Loading images", end='\r', flush=True)
    content_title, content_image, style_images = load_style_content_imgs(style_links, content_link)
    print("Images loaded.")


    # Define content and style representations
    content_layers: List[str] = [
        'block5_conv2'
    ] 

    style_layers: List[str] = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1', 
        'block4_conv1', 
        'block5_conv1'
    ]

    # Cycle through the hyperparameters:
    for style_weight in style_weights:
        for content_weight in content_weights:
            for total_variance_weight in total_variance_weights:
                # Initialize the model:
                print("Initializing model", end='\r', flush=True)
                model = StyleTransfer(style_layers, content_layers, style_images, content_image, style_weight, content_weight, total_variance_weight)
                print(f"Model initialized {style_weight=}, {content_weight=}, {total_variance_weight=}")

                # Train the model:
                print("\nTraining model:")
                img = model.train(epochs=10, visuals=visuals)    
                print("\nModel trained.")

                # Save the output image:
                output_file_path = f"basic/output/search/{content_title}_{style_weight}_{content_weight}_{total_variance_weight}_{int(time.time())}.jpg"
                print("Saving the output image", end='\r', flush=True)
                with open(output_file_path,'wb') as f:
                    tensor_to_image(img).save(f, "JPEG")
                print(f"Output image saved at {output_file_path}")

def conv_layer_search(
    style_links: Dict[str, str], 
    content_link: str, 
    content_layers: List[str],
    style_layers: List[str],
    visuals: bool = False) -> None:
    """
    Performs style transfer with different hyperparameters from the style images to the content image.

    Parameters:
    style_links (Dict[str, str]): A dictionary where keys are names of the style images and values are their URLs.
    content_link (str): The URL of the content image.
    style_weights (List[float]): List of style weights to try.
    content_weights (List[float]): List of content weights to try.
    total_variance_weights (List[float]): List of total variance weights to try.
    visuals (bool, optional): If True, visualizes the training process. Defaults to False.
    """
    # Load the content and style images:
    print("Loading images", end='\r', flush=True)
    content_title, content_image, style_images = load_style_content_imgs(style_links, content_link)
    print("Images loaded.")

    content_layer_combos = powerset(content_layers)
    print(content_layer_combos)
    # style_layer_combos = powerset(style_layers)
    style_layer_combos = {'11111' : style_layers}
    
    # Cycle through the hyperparameters:
    for content_layer_id, content_layer_combo in content_layer_combos.items():
        for style_layer_id, style_layer_combo in style_layer_combos.items():
            print("Initializing model", end='\r', flush=True)
            model = StyleTransfer(style_layer_combo, content_layer_combo, style_images, content_image)
            print(f"Model initialized {content_layer_combo=}, {style_layer_combo=}")
            print("\nTraining model:")
            img = model.train(epochs=10, visuals=visuals)    
            print("\nModel trained.")
            output_file_path = f"basic/output/search/{content_title}_{content_layer_id}_{style_layer_id}_{int(time.time())}.jpg"
            print("Saving the output image", end='\r', flush=True)
            with open(output_file_path,'wb') as f:
                tensor_to_image(img).save(f, "JPEG")
            print(f"Output image saved at {output_file_path}")

def main():
    style_links = {
        'Gwen1': 'https://static.wikia.nocookie.net/p__/images/2/2d/Gwenhugsherdad.jpg/revision/latest/scale-to-width-down/1000?cb=20230831234851&path-prefix=protagonist',
        'Gwen2': 'https://static.wikia.nocookie.net/intothespiderverse/images/1/12/Gwenleavesband.jpg/revision/latest/scale-to-width-down/1000?cb=20230927002710',
        'Gwen3': 'https://static.wikia.nocookie.net/intothespiderverse/images/1/1b/Rippeter.jpg/revision/latest/scale-to-width-down/1000?cb=20230927003040',
        'Gwen4': 'https://static.wikia.nocookie.net/intothespiderverse/images/d/d9/Fightingthelizard.jpg/revision/latest/scale-to-width-down/1000?cb=20230928231607',
        'Gwen5': 'https://static.wikia.nocookie.net/intothespiderverse/images/f/f7/Gwenpeter.jpg/revision/latest/scale-to-width-down/1000?cb=20230927002716'
    }
    content_link = {
        'MonaLisa': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg'
        # 'Gwen3': 'https://static.wikia.nocookie.net/intothespiderverse/images/1/1b/Rippeter.jpg/revision/latest/scale-to-width-down/1000?cb=20230927003040'
    }

    style_transfer(style_links, content_link, True)
    # hyperparameter_search(style_links, content_link, [1e-1, 1e-2, 1e-3], [1e4, 1e5, 1e6], [10, 20, 30, 40, 50], False)
    # conv_layer_search(style_links, content_link, ['block4_conv1', 'block4_conv2', 'block5_conv1', 'block5_conv2'], ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'], False)
  
if __name__ == "__main__":
    main()
