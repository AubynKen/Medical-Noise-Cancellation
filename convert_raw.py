import PIL
import os
from PIL import Image


def convert(format: str, path_to_data_dir="./data", img_size=(1024, 1024)):
    raw_dir = os.path.join(path_to_data_dir, "raw")
    format_dir = os.path.join(path_to_data_dir, format)

    # if os.path.isdir(format_dir):
    #     print(f"{format_dir} already exists")
    #     return  # early exit

    # os.mkdir(format_dir)
    for filename in os.listdir(raw_dir):
        raw_path = os.path.join(raw_dir, filename)
        target_path = os.path.join(format_dir, f"{filename.split('.')[0]}.{format}")
        print(target_path)
        raw = open(raw_path, "rb")
        raw_data = raw.read()
        raw.close()
        img = Image.frombytes('L', img_size, raw_data, "raw") # , 'F;16')
        # print(raw_path)
        # print(target_path)
        img.save(target_path)


convert("png")
