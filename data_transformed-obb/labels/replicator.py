# script that copies txt files to a new folder and gives them new names

import os

suffixes = ['_w', '_h']
folders = ['train', 'test', 'val']
output_folder = 'resized'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for folder in folders:
    files = os.listdir(folder)

    if not os.path.exists(os.path.join(output_folder, folder)):
        os.makedirs(os.path.join(output_folder, folder))

    for file in files:
        filename, file_extension = os.path.splitext(file)

        for suffix in suffixes:
            new_file_name = filename + suffix + file_extension
            with open(os.path.join(folder, file), 'r') as f:
                with open(os.path.join(output_folder, folder, new_file_name), 'w') as new_f:
                    new_f.write(f.read())