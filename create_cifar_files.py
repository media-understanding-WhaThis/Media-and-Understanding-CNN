"""

Script to create CIFAR like data files

"""

import glob
import logging
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import progressbar
from PIL import Image

logging.basicConfig(level=logging.DEBUG)


class ImageProcessor:
    def __init__(self, image_path):
        """
        Image processor helper class

        :param image_path: path to an image
        """
        self.im = Image.open(image_path)
        self.im = self.im.convert('RGB')

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

        self.im = self.im.crop((int(left), int(top), int(right), int(bottom)))

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


def write_data_to_dict(out_file_name, images_file_paths, labels, batch_label, max_images):
    """
    Creates a dict with the following keys ['filenames', 'labels', 'data', 'batch_label']

    The data entry is a <#images>x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The
    first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The
    image is stored in row-major order, so that the first 32 entries of the array are the red channel values of
    the first row of the image.

    https://www.cs.toronto.edu/~kriz/cifar.html

    :param out_file_name: string file name of the binary
    :param images_file_paths: list of paths to image files
    :param labels: list labels for each file
    :param batch_label: label of the whole dictionary
    """
    data = dict()
    data['labels'] = []
    data['data'] = []

    if len(images_file_paths) < max_images:
        logging.error('Not enough images provided for {0}, provided {1} but max is set to {2}'.format(
            batch_label,
            len(images_file_paths),
            max_images)
        )
        return

    bar = progressbar.ProgressBar(max_value=len(images_file_paths))
    for idx, image_file_path in enumerate(images_file_paths):
        path = Path(image_file_path)
        if path.exists():
            if idx is max_images:
                break

            data['filenames'] = path.name
            data['labels'].append(labels[idx])
            data['batch_label'] = batch_label

            processor = ImageProcessor(path)
            processor.center_crop()
            processor.resize_image()
            data['data'].append(processor.create_int8_array())
            bar.update(idx)
        else:
            print('Could not find path {}'.format(path))

    data['data'] = np.asarray(data['data'], dtype=np.uint8)

    pickle.dump(data, open(out_file_name, 'wb'))
    print(' Dumped dictionary to file {}'.format(out_file_name))


def write_data_to_bin(out_file_name, images_file_paths, labels):
    """
    Writes binary for all images and associated labels. Index
    of image is index of label.

    The first byte is the label of the first image, which is a number in the range 0-9. The next 3072 bytes are the
    values of the pixels of the image. The first 1024 bytes are the red channel values, the next 1024 the green, and
    the final 1024 the blue. The values are stored in row-major order, so the first 32 bytes are the red channel
    values of the first row of the image.

    :param out_file_name: string file name of the binary
    :param image_file_paths: list of paths to image files
    :param labels: list labels for each file
    """

    with open(out_file_name, 'wb') as f:
        for file_path, label in zip(images_file_paths, labels):
            # Process image
            processor = ImageProcessor(file_path)
            processor.center_crop()
            processor.resize_image()
            arr = processor.create_int8_array()

            # Add label
            label_arr = np.asarray([label], dtype=np.uint8)
            arr = np.concatenate((label_arr, arr), axis=0)

            # Write bytes to file
            f.write(arr.tobytes())


if __name__ == '__main__':
    # Add label to this dictionary
    labels = {
        'rose': 0,
        'sunflower': 1,
        'daisy': 2
    }
    
    train_size = 600
    test_size = 100
    
    # ROSES
    rose_all_filenames = glob.glob('data/plantset/rose/*')
    print('There are {} rose files.'.format(len(rose_all_filenames)))
    # rose_train_files = roses_all_filenames[:int(len(roses_all_filenames) * .90)]  # Percentage
    # rose_test_files = roses_all_filenames[-int(len(roses_all_filenames) * .10):]
    rose_train_files = rose_all_filenames[:train_size]  # TODO FIX set real data size
    rose_test_files = rose_all_filenames[-test_size:]  # TODO FIX set real data size
    print('Roses train file...')
    write_data_to_dict('data/plantset/train_rose.p', rose_train_files, [labels['rose']] * len(rose_train_files),
                       'rose_train', max_images=train_size)
    print('Roses test file...')
    write_data_to_dict('data/plantset/test_rose.p', rose_test_files, [labels['rose']] * len(rose_test_files),
                       'rose_test', max_images=test_size)

    # SUNFLOWERS
    sunflower_all_filenames = glob.glob('data/plantset/sunflower/*')
    print('There are {} sunflower files.'.format(len(sunflower_all_filenames)))
    # sunflower_train_files = sunflowers_all_filenames[:int(len(sunflowers_all_filenames) * .90)]  # Percentage
    # sunflower_test_files = sunflowers_all_filenames[-int(len(sunflowers_all_filenames) * .10):]
    sunflower_train_files = sunflower_all_filenames[:train_size]  # TODO FIX set real data size
    sunflower_test_files = sunflower_all_filenames[-test_size:]  # TODO FIX set real data size
    print('Sunflowers train file...')
    write_data_to_dict('data/plantset/train_sunflower.p', sunflower_train_files,
                       [labels['sunflower']] * len(sunflower_train_files),
                       'sunflower_train', max_images=train_size)
    print('Sunflowers test file...')
    write_data_to_dict('data/plantset/test_sunflower.p', sunflower_test_files,
                       [labels['sunflower']] * len(sunflower_test_files),
                       'sunflower_test', test_size)

    # COMMON DAISIES
    daisy_all_filenames = glob.glob('data/plantset/daisy/*')
    print('There are {} daisy files.'.format(len(daisy_all_filenames)))
    # daisy_train_files = daisy_all_filenames[:int(len(daisy_all_filenames) * .90)]  # Percentage
    # daisy_test_files = daisy_all_filenames[-int(len(daisy_all_filenames) * .10):]
    daisy_train_files = daisy_all_filenames[:train_size]  # TODO FIX set real data size
    daisy_test_files = daisy_all_filenames[-test_size:]  # TODO FIX set real data size
    print('Daisy train file...')
    write_data_to_dict('data/plantset/train_daisy.p', daisy_train_files, [labels['daisy']] * len(daisy_train_files),
                       'daisy_train', max_images=train_size)
    print('Daisy test file...')
    write_data_to_dict('data/plantset/test_daisy.p', daisy_test_files, [labels['daisy']] * len(daisy_test_files),
                       'daisy_test', max_images=test_size)
