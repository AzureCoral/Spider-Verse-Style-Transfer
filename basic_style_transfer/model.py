import os
import tensorflow as tf
from typing import List, Dict, Union

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

def clip_0_1(image: tf.Tensor) -> tf.Tensor:
  """
  Clips tensor values between 0 and 1.

  Parameters:
  image (tf.Tensor): The tensor to clip.

  Returns:
  tf.Tensor: The clipped tensor.
  """
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def vgg_layers(layer_names: List[str]) -> tf.keras.Model:
  """
  Creates a VGG model that returns a list of intermediate output values.

  Parameters:
  layer_names (List[str]): The names of the layers to include in the model.

  Returns:
  tf.keras.Model: The resulting VGG model.
  """
  # Load our model. Load pretrained VGG, trained on ImageNet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
  """
  Calculates the Gram matrix for a given tensor.

  Parameters:
  input_tensor (tf.Tensor): The tensor to calculate the Gram matrix of.

  Returns:
  tf.Tensor: The resulting Gram matrix.
  """
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers: List[str], content_layers: List[str]):
    """
    Initializes the StyleContentModel.

    Parameters:
    style_layers (List[str]): The names of the style layers to include in the model.
    content_layers (List[str]): The names of the content layers to include in the model.
    """
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs: tf.Tensor) -> Dict[str, Dict[str, tf.Tensor]]:
    """
    Calls the StyleContentModel.

    Parameters:
    inputs (tf.Tensor): The input tensor.

    Returns:
    Dict[str, Dict[str, tf.Tensor]]: A dictionary containing the style and content outputs.
    """
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

class StyleTransfer():
  def __init__(self, style_layers: List[str], content_layers: List[str], style_image: tf.Tensor, content_image: tf.Tensor):
    """
    Initializes the StyleTransfer.

    Parameters:
    style_layers (List[str]): The names of the style layers to include in the model.
    content_layers (List[str]): The names of the content layers to include in the model.
    style_image (tf.Tensor): The style image tensor.
    content_image (tf.Tensor): The content image tensor.
    """
    self.extractor = StyleContentModel(style_layers, content_layers)
    self.style_targets = self.extractor(style_image)['style']
    self.content_targets = self.extractor(content_image)['content']
    self.opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    self.image = tf.Variable(content_image)

  def style_content_loss(self, outputs: Dict[str, Dict[str, tf.Tensor]], style_weight: float =1e-2, content_weight: float = 1e4) -> tf.Tensor:
    """
    Calculates the style and content loss.

    Parameters:
    outputs (Dict[str, Dict[str, tf.Tensor]]): The style and content outputs.
    style_weight (float): The weight of the style loss.
    content_weight (float): The weight of the content loss.

    Returns:
    tf.Tensor: The total loss.
    """
    num_style_layers = len(self.style_targets)
    num_content_layers = len(self.content_targets)

    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name])**2) 
                          for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name])**2) 
                              for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss
  
  @tf.function()
  def train_step(self, image: tf.Tensor, total_variation_weight: float = 30) -> None:
    """
    Performs one training step.

    Parameters:
    image (tf.Tensor): The input image tensor.
    total_variation_weight (float): The weight of the total variation loss.
    """
    with tf.GradientTape() as tape:
        outputs = self.extractor(image)
        loss = self.style_content_loss(outputs)
        loss += total_variation_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    self.opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

  def train(self, epochs: int = 10, steps_per_epoch: int = 100) -> tf.Tensor:
    """
    Trains the model for a specified number of epochs.

    Parameters:
    epochs (int): The number of epochs to train for.
    steps_per_epoch (int): The number of steps per epoch.

    Returns:
    tf.Tensor: The final image tensor after training.
    """
    for epoch in range(epochs):
      print(f"Epoch: {epoch+1} ", end="")
      for _ in range(steps_per_epoch):
          self.train_step(self.image)
          print(".", end='', flush=True)
      print("")
    return self.image