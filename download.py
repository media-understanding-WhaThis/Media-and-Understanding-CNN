from urllib import request
from urllib import error

with open('rose_urls.txt', 'r') as file:
    for line in file:
        file_name = line.split('/')[-1].strip('\n')
        try:
            request.urlretrieve(line, 'rose/' + file_name)
        except ConnectionResetError as e:
            print('File failed connection error: {0}'.format(file_name))
            pass
        except error.HTTPError:
            print('File failed http error: {0}'.format(file_name))