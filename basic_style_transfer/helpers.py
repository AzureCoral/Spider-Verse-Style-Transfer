import tensorflow as tf
import numpy as np
import PIL.Image

def tensor_to_image(tensor: np.ndarray) -> PIL.Image.Image:
  """
  Converts a tensor into an image.

  Parameters:
  tensor (np.ndarray): The tensor to convert.

  Returns:
  PIL.Image.Image: The resulting image.
  """
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img: str) -> tf.Tensor:
  """
  Loads an image from a file, converts it to a tensor, and resizes it to have a maximum dimension of 512.

  Parameters:
  path_to_img (str): The path to the image file.

  Returns:
  tf.Tensor: The resulting tensor.
  """
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img