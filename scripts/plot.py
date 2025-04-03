### script for producing plots ###
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataloaders import mnist_loader
from utils import config_loader

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/plot.yaml')
    return parser.parse_args()

def plot_image(image, 
               title=None,
               output_dir=None,
               ):
    """Plot a single image."""
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
    if output_dir:
        plt.savefig(os.path.join(output_dir, title + '.png'))

def match_images(im, im_set):
    """
    Match an image with another image of the same label in a given set.
    Args:
        im: The image to be matched.
        im_set: The set of images to search in.
    Returns:
        matched_image: The matched image from the set.
    """
    for image in im_set:
        if image[1] == im[1]:
            return image
    print("No matching image found.")
    return None

def unmatched_images(im, im_set):
    """
    Get an image that does not match the label of the given image.
    Args:
        im: The image to be matched.
        im_set: The set of images to search in.
    Returns:
        unmatched_image: An unmatched image from the set.
    """
    for image in im_set:
        if image[1] != im[1]:
            return image
    print("No unmatched image found.")
    return None

def main():
    args = argparser()
    config = config_loader(args.config)

    ### create output directory
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### get some images ###
    train_loader = mnist_loader(batch=1, train=True)
    images = train_loader.get_set(100)

    ### get two images with same label ###
    print(images)
    im0 = images[0]
    ### remove im0 from the set to avoid matching with itself ###
    images.remove(im0)
    label0 = images[1]
    im1 = match_images(im0, images)
    im2 = unmatched_images(im0, images)

    plot_image(im0[0].numpy(), title='Image_0', output_dir=output_dir)
    plot_image(im1[0].numpy(), title='Image_1', output_dir=output_dir)
    plot_image(im2[0].numpy(), title='Image_2', output_dir=output_dir)
    print(f"Image 0 label: {im0[1]}")
    print(f"Image 1 label: {im1[1]}")
    print(f"Image 2 label: {im2[1]}")

    ### plot concatenation and addition of images ###
    im0 = im0[0].numpy()
    im1 = im1[0].numpy()
    im2 = im2[0].numpy()
    im01_concat = np.concatenate((im0, im1), axis=0)
    im01_add = im0 + im1
    im02_concat = np.concatenate((im0, im2), axis=0)
    im02_add = im0 + im2

    plot_image(im01_concat, title='Image_0_concat_Image_1', output_dir=output_dir)
    plot_image(im01_add, title='Image_0_add_Image_1', output_dir=output_dir)
    plot_image(im02_concat, title='Image_0_concat_Image_2', output_dir=output_dir)
    plot_image(im02_add, title='Image_0_add_Image_2', output_dir=output_dir)



if __name__ == "__main__":
    main()