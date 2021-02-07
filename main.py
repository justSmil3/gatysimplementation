import os
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import time
import functools
import tensorflow_hub as hub

def setup():
    os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
    mpl.rcParams['figure.figsize'] = (12,12)
    mpl.rcParams['axes.grid'] = False

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
    
def load_img(path):
    max_dim = 512
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def fastmodel(content_path, style_path):
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    tensor_to_image(stylized_image)



    x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    prediction_probabilities = vgg(x)

    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    
def saveimage(image, name):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    img = tf.keras.preprocessing.image.array_to_img(image)
    img.save(str(name) + ".png")
   

def prep_vgg():

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    print()
    for layer in vgg.layers:
        print(layer.name)
    

    #from gatys et al
    content_layers = ['block5_conv2'] 

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    return content_layers, style_layers

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name:value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content':content_dict, 'style':style_dict}

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs, 
                       style_weight, 
                       content_weight, 
                       num_style_layers, 
                       num_content_layers, 
                       content_targets,
                       style_targets):
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])

    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])

    content_loss *= content_weight / num_content_layers

    loss = content_loss + style_loss
    return loss


def prep_train(content_image, style_image, extractor):
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)
    opt = tf.optimizers.Adam(learning_rate=0.0002, beta_1=0.99, epsilon=1e-1)

    style_weight = 1e-1
    content_weight = 1e-1
    total_variation_weight = 0.1
    return image, opt, style_weight, content_weight, total_variation_weight, content_targets, style_targets

def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]

    return x_var, y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


@tf.function()
def train_step(opt, 
               image, 
               extractor,  
               style_weight, 
               content_weight,
               total_variation_weight, 
               num_style_layers, 
               num_content_layers,
               content_targets,
               style_targets):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, 
                                  style_weight, 
                                  content_weight, 
                                  num_style_layers, 
                                  num_content_layers,
                                  content_targets,
                                  style_targets)
        loss += total_variation_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

#content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
#style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

content_prefix = "./content/"
style_prefix = "./style/"

contentfiles = [f for f in os.listdir("./content/")]
stylefiles = [f for f in os.listdir("./style/")]

s = 2
c = 0

content_img_name = "muster.png"
style_img_name = "WCTstyleCustom.jpg"

content_name  = content_img_name.split('.')[0]
style_name = style_img_name.split('.')[0]
img_name = content_name + "-" + style_name

content_path = content_prefix + content_img_name
style_path = style_prefix + style_img_name

content_image = load_img(content_path)
style_image = load_img(style_path)

content_layers, style_layers = prep_vgg()
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

content_image = load_img(content_path)
style_image = load_img(style_path)

extractor = StyleContentModel(style_layers, content_layers)

image, opt, style_weight, content_weight, total_variation_weight, content_targets, style_targets = \
    prep_train(content_image, style_image, extractor)



start = time.time()

epochs = 150
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(opt, 
               image, 
               extractor, 
               style_weight, 
               content_weight,
               total_variation_weight, 
               num_style_layers, 
               num_content_layers,
               content_targets,
               style_targets)
    print(".", end='')
  display.clear_output(wait=True)
  display.display(tensor_to_image(image))
  print("Train step: {}".format(step))
  saveimage(image, img_name)
saveimage(image, img_name)
end = time.time()
print("Total time: {:.1f}".format(end-start))



