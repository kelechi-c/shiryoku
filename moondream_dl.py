from typing import List
import urllib
import requests
import os
import pandas as pd
import cv2
from PIL import Image as pillow_image
from multiprocessing import Pool
from datasets import load_dataset
from tqdm.auto import tqdm

def load_image(url):
    try:
        url_content = requests.get(url, stream=True).raw
        image = pillow_image.open(url_content)
        return image

    except Exception as e:

        print(f"Error loading image: {url}, {e}")
        return None

sample_image = "ignore/images/laptoppic.jpeg"


def cv2play(image_file):
    image = cv2.imread(image_file)

    cv2.imshow("Laptop", image)
    cv2.waitKey(0)
    
    return image

cv2play(sample_image)


# Download single images
image_folder = "moondream_images"
out_folder = os.path.join(os.getcwd(), image_folder)

def download_img(url):
    try:
        response_file = requests.get(url)
        out_file = os.path.basename(url)
        out_path = os.path.join(out_folder, out_file)

        with open(out_path, "wb") as image_file:
            image_file.write(response_file.content)
            yield image_file

            print(f"{out_file} downloaded")

    except Exception as e:
        print(f"error: {e}")


def get_moondream_data():
    moondream_dataset = load_dataset("isidentical/moondream2-coyo-5M-captions")
    md_data = moondream_dataset["train"][:100000]  # type: ignore
    image_urls = md_data["url"]  # type: ignore
    descriptions = md_data["moondream2_caption"]  # type: ignore

    count = 0

    for url, desc in tqdm(zip(image_urls, descriptions)):
        image_dl = download_img(url)
        caption = desc.lower()

        count += 1

        if image_dl is not None:
            print(f"image no.{count}")
            yield (image_dl, image_dl.name, caption) # type: ignore


q, k, v = zip(*get_moondream_data())


def moondream_csv(path: List, url: List, desc: List, output_path: str = os.getcwd()):
    print("Writing to csv..")

    keys = ["image_path", "image_url", "caption"]
    md_dict = dict(zip(keys, zip(path, url, desc)))
    moondream_df = pd.DataFrame(md_dict)

    moondream_df.to_csv("moondream2.csv")

    print("Csv transfer complete...")


moondream_csv(q, k, v)