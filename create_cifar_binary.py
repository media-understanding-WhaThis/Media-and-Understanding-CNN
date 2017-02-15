__author__ = "Yorick de Boer"

from PIL import Image
from itertools import product
import numpy as np

class ImageProcessor:
    def __init__(self, image_path):
        """
        Image processor helper class

        :param image_path: path to an image
        """
        self.im = Image.open(image_path)
        self.im.convert('RGB')

    def center_crop(self):
        """
        Center crops an image to rectangle, determined by the side with the largest size.

        :return: Image object
        """

        current_width, current_height = self.im.size

        if current_width > current_height:
            left = (current_width - current_height) / 2
            right = (current_width + current_height) / 2
            top, bottom = 0, current_height
        else:
            top = (current_height - current_width) / 2
            bottom = (current_height + current_width) / 2
            left, right = 0, current_width

        self.im = self.im.crop((left, top, right, bottom))

    def resize_image(self):
        """
        Resizes image to specific size

        :return: Image object
        """

        self.im = self.im.resize((32, 32), Image.BICUBIC)

    def save_image(self, file_path):
        self.im.save(file_path)

    def create_int8_array(self):
        """
        Create an int8 byte stream of RGB values, 3 x 1024 bytes.

        :return: byte stream
        """

        arr = np.asarray(self.im, dtype=np.uint8)

        # Pillow use column-major notation while numpy uses row-major notation
        arr = np.transpose(arr, (1, 0, 2))

        red = arr[:, :, [0]].flatten()
        green = arr[:, :, [1]].flatten()
        blue = arr[:, :, [2]].flatten()

        # Concatenate all arrays behind each other in the order of R G B
        arr = np.concatenate((red, green, blue), axis=0)

        # arr = np.vstack((red, green, blue))

        return arr

    def iterator_pixel_color(self):
        """
        Pixel color generator

        :return:
        """

        x, y = self.im.size
        for x, y in product(range(x), range(y)):
            yield self.im.getpixel((x, y))


def create_labeled_array(np_arr, label):
    label_arr = np.asarray([label], dtype=np.uint8)
    return np.concatenate((label_arr, np_arr), axis=0)


def write_images_bytes_to_file(out_file_name, image_file_paths, labels):
    """
    Writes binary for all images and associated labels. Index
    of image is index of label.

    :param out_file_name: string file name of the binary
    :param image_file_paths: list of paths to image files
    :param labels: list labels for each file
    :return:
    """

    with open(out_file_name, 'wb') as f:
        for file_path, label in zip(image_file_paths, labels):
            # Process image
            processor = ImageProcessor(file_path)
            processor.center_crop()
            processor.resize_image()
            arr = processor.create_int8_array()

            # Add label
            arr = create_labeled_array(arr, label)

            # Write bytes to file
            f.write(arr.tobytes())


if __name__ == '__main__':
    # np.set_printoptions(threshold=np.nan)

    write_images_bytes_to_file(
        'plant_binary.bin',
        ['/home/y/Media and Understanding/cifar10/test_image.jpg'],
        [1]
    )
