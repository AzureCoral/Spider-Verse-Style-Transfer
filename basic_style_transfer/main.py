import tensorflow as tf
from helpers import *
from model import StyleContentModel

from PIL import Image

def main():
    # Load the content and style images:
    print("Loading images", end='\r', flush=True)
    style_path = tf.keras.utils.get_file('Gwen.jpg', 'https://static.wikia.nocookie.net/p__/images/2/2d/Gwenhugsherdad.jpg/revision/latest/scale-to-width-down/1000?cb=20230831234851&path-prefix=protagonist')
    content_path = tf.keras.utils.get_file('Mona_lisa.jpg','https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg')

    content_image = load_img(content_path)
    style_image = load_img(style_path)
    print("Images loaded.")

    # Define content and style representations
    print("Defining content and style representations", end='\r', flush=True)
    content_layers = ['block5_conv2'] 

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    print("Content and style representations defined.")

    # Initialize the model:
    print("Initializing model", end='\r', flush=True)
    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    print("Model initialized.")

    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def style_content_loss(outputs, style_weight=1e-2, content_weight=1e4):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) 
                            for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                                for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(image, total_variation_weight=30):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            loss += total_variation_weight*tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    # Initalize the image to the content image:
    print("Initializing image", end='\r', flush=True)
    image = tf.Variable(content_image)
    print("Image initialized.")

    # Train the model:
    print("\nTraining model:")
    epochs = 5
    steps_per_epoch = 100

    step = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1} ", end="")
        for _ in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end='', flush=True)
        print("")
        
    print("\nModel trained.")

    # Save the output image:
    output_file_path = "output.jpg"
    print("Saving the output image", end='\r', flush=True)
    with open(output_file_path,'wb') as f:
        tensor_to_image(image).save(f, "JPEG")
    print(f"Output image saved at {output_file_path}")
    
main()