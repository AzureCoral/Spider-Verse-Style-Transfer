import tensorflow as tf
from helpers import *
from model import StyleTransfer
from typing import Dict, List

def style_transfer(style_links: Dict[str, str], content_link: str, visuals: bool = False) -> None:
    """
    Performs style transfer from the style images to the content image.

    Parameters:
    style_links (Dict[str, str]): A dictionary where keys are names of the style images and values are their URLs.
    content_link (str): The URL of the content image.
    visuals (bool, optional): If True, visualizes the training process. Defaults to False.
    """
    # Load the content and style images:
    print("Loading images", end='\r', flush=True)
    content_path = tf.keras.utils.get_file('content', content_link)
    style_paths = [tf.keras.utils.get_file(key, style_links[key]) for key in style_links]
    content_image = load_img(content_path)
    style_images = [load_img(style_path) for style_path in style_paths]
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
    model = StyleTransfer(style_layers, content_layers, style_images, content_image)
    print("Model initialized.")

    # Train the model:
    print("\nTraining model:")
    img = model.train(visuals=visuals)    
    print("\nModel trained.")

    # Save the output image:
    output_file_path = "results/unbatched/basic_gwen.jpg"
    print("Saving the output image", end='\r', flush=True)
    with open(output_file_path,'wb') as f:
        tensor_to_image(img).save(f, "JPEG")
    print(f"Output image saved at {output_file_path}")

def main():
    # style_links = {
    #     'Gwen1': 'https://static.wikia.nocookie.net/p__/images/2/2d/Gwenhugsherdad.jpg/revision/latest/scale-to-width-down/1000?cb=20230831234851&path-prefix=protagonist',
    #     'Gwen2': 'https://static.wikia.nocookie.net/p__/images/2/2d/Gwenhugsherdad.jpg/revision/latest/scale-to-width-down/1000?cb=20230831234851&path-prefix=protagonist'
    # }
    # content_link = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg'
    
    # style_transfer(style_links, content_link, True)

    style_links = {
        'Gwen1': 'https://static.wikia.nocookie.net/p__/images/2/2d/Gwenhugsherdad.jpg/revision/latest/scale-to-width-down/1000?cb=20230831234851&path-prefix=protagonist'
    }
    content_link = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg'
    
    style_transfer(style_links, content_link, True)
    
if __name__ == "__main__":
    main()
