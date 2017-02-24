from urllib import request
from urllib import error
from hashlib import md5
from glob import glob
from os import remove

DIRECTORY = 'rose/'


def download():
    with open('rose_urls.txt', 'r') as file:
        for line in file:
            file_name = line.split('/')[-1].strip('\n')
            try:
                request.urlretrieve(line, DIRECTORY + file_name)
            except ConnectionResetError as e:
                print('File failed connection error: {0}'.format(file_name))
                pass
            except error.HTTPError:
                print('File failed http error: {0}'.format(file_name))


def find_duplicate_hashes():
    files = glob(DIRECTORY + '*')
    return [md5(open(file, 'rb').read()).hexdigest() for file in files]


def list_duplicate_files(file_md5):
    files = glob(DIRECTORY + '*')
    return [file for file in files if md5(open(file, 'rb').read()).hexdigest() == file_md5]


def remove_files(files):
    for file in files:
        print('Removing file: {0}'.format(file))
        remove(file)


if __name__ == '__main__':
    error_example = md5(open(DIRECTORY + '2514213891_5ccff5260b.jpg', 'rb').read()).hexdigest()
    remove_files(list_duplicate_files(error_example))
