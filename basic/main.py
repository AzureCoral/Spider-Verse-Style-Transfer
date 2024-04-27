import tensorflow as tf
from helpers import *
from model import StyleTransfer

def main():
    # Load the content and style images:
    print("Loading images", end='\r', flush=True)
    style_path = tf.keras.utils.get_file('Gwen.jpg', 'https://static.wikia.nocookie.net/p__/images/2/2d/Gwenhugsherdad.jpg/revision/latest/scale-to-width-down/1000?cb=20230831234851&path-prefix=protagonist')
    content_path = tf.keras.utils.get_file('Mona_lisa.jpg','https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg')

    content_image = load_img(content_path)
    style_image = load_img(style_path)
    print("Images loaded.")

    # Define content and style representations
    content_layers = ['block5_conv2'] 

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    # Initialize the model:
    print("Initializing model", end='\r', flush=True)
    model = StyleTransfer(style_layers, content_layers, style_image, content_image)
    print("Model initialized.")

    # Train the model:
    print("\nTraining model:")
    img = model.train(visuals=True)    
    print("\nModel trained.")

    # Save the output image:
    output_file_path = "outputs/basic_gwen.jpg"
    print("Saving the output image", end='\r', flush=True)
    with open(output_file_path,'wb') as f:
        tensor_to_image(img).save(f, "JPEG")
    print(f"Output image saved at {output_file_path}")

if __name__ == "__main__":
    main()