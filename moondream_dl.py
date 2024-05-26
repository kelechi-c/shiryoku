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


def load_image_wrapper(url):
    try:
        image = download_img(url)
        return image
    except:
        return None


def get_moondream_data():
    moondream_dataset = load_dataset("isidentical/moondream2-coyo-5M-captions")
    md_data = moondream_dataset["train"][:10]  # type: ignore
    image_urls = md_data["url"]  # type: ignore
    descriptions = md_data["moondream2_caption"]  # type: ignore

    with Pool(processes=7) as pool:
        images = pool.imap(load_image_wrapper, tqdm(image_urls))
        captions = [desc.lower() for desc in descriptions]
        

    for image, caption in tqdm(zip(images, captions)):
        if image is not None:
            yield (image_urls, image, caption)
            
    
        

def moondream_csv(dataframe: pd.DataFrame, output_path: str):
    print('Writing to csv..')
    dataframe.to_csv(output_path)
    print('Csv transfer complete...')
    
    