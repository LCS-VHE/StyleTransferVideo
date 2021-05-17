import tensorflow_hub as hub
import tensorflow as tf
import time # Debugging
import os 
from PIL import Image
import numpy as np


def mkdir(path):
    try:
        os.mkdir(path)
    except Exception as e:
        pass
    
    
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img = img[tf.newaxis, :]
    return img


def load_imgs(paths, style_img_path):
    ori_imgs, style_imgs = tf.constant(load_img(paths[0])), tf.constant(load_img(style_img_path))

    for path in paths[1:]:
        ori_imgs = tf.concat([ori_imgs, load_img(path)], axis=0)
        style_imgs = tf.concat([style_imgs, load_img(style_img_path)], axis=0)
        
    return ori_imgs, style_imgs
        

def save_imgs(imgs, count, save_dir):
    count = count
    for image in imgs:
        image = np.array(image.numpy() * 255, dtype=np.uint8)
        Image.fromarray(image).save(os.path.join(save_dir, f"frame {count}.png"))
        count += 1 

        
def stylize_images(source_dir, style_img_path, save_dir=os.path.join(os.curdir, "Stylized_Image"), batch_size=10):
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    imgs_path = [os.path.join(source_dir, f"frame {count}.jpg") for count in range(len(os.listdir(source_dir)))]
    mkdir(save_dir)
    
    for i in range(int(len(os.listdir(source_dir))/batch_size)):
        if i == len(os.listdir(save_dir)) -1:
            ori_imgs, style_img = load_imgs(imgs_path[i*batch_size:], style_img_path)
        else:
            ori_imgs, style_img = load_imgs(imgs_path[i*batch_size:(i+1)*batch_size], style_img_path)

        start = time.time()
        stylized_image = model(ori_imgs, style_img)[0]
        print(time.time() - start)
        print(i)
        save_imgs(stylized_image, i*batch_size, save_dir)
    
print("Start")
stylize_images(os.path.join(os.curdir, "Image"), 'style.jpg')