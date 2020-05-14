from __future__ import print_function

import os
from pdf2image import convert_from_path
import cv2
import numpy as np

def get_filenames(path, dpi=500):
    if not os.path.exists(path):
        os.makedirs(path)
    elif len(os.listdir(path)) == 0:
        print("No file in this folder")
        return 0

    # Rename files with '.' to '_'
    doc_paths = []
    for filename in os.listdir(path):
        if filename.endswith('.pdf'):
            new_filename = filename.replace('.', '_')
            new_filename = new_filename.replace('_pdf', '.pdf')
            os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
            doc_paths.append(os.path.join(path, new_filename))

    print('Converting pdfs to images...')

    # Convert PDF to JPEG
    # image_paths = []
    # for filename in doc_paths:
    #     if filename.endswith('.pdf'):
    #         pages = convert_from_path(filename, dpi=dpi)

    #         for index, page in enumerate(pages):
    #             image_filename = filename.replace('.pdf', '') + '-' + str(index+1) + '.jpg'
    #             image_paths.append(image_filename)
    #             page.save(image_filename, 'JPEG')

    # ... or run this if no conversion is required
    image_paths = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            image_paths.append(os.path.join(path, filename))

    print('Done converting pdfs...')

    # 90% train images and 10% test images
    n_train_samples = int(len(image_paths) * 0.9)
    train_filenames = image_paths[:n_train_samples]
    test_filenames = image_paths[n_train_samples:]

    # added to work with BINARY IMAGE
    train_image_array = []
    for image in train_filenames:
        im = cv2.imread(image)
        im = cv2.resize(im, (160,160))          # (224,224)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        train_image_array.append(np.asarray(im)) #.transpose(1, 0, 2))
    
    test_image_array = []
    for image in test_filenames:
        im = cv2.imread(image)
        im = cv2.resize(im, (160,160))          # (224,224)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        test_image_array.append(np.asarray(im)) #.transpose(1, 0, 2))

    train_image_array = np.array(train_image_array)
    test_image_array = np.array(test_image_array)

    # return train_filenames, test_filenames
    return train_image_array, test_image_array

