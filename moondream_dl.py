from typing import List
import io
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


def get_moondream_data(split_size: int):
    moondream_dataset = load_dataset("isidentical/moondream2-coyo-5M-captions")
    md_data = moondream_dataset["train"][:split_size]  # type: ignore
    image_urls = md_data["url"]  # type: ignore
    descriptions = md_data["moondream2_caption"]  # type: ignore

    count = 0

    for url, desc in tqdm(zip(image_urls, descriptions)):
        url = str(url)
        if url.endswith(("jpeg", "jpg", "png")):

            image_dl = download_img(url)
            caption = desc.lower()
            file_name = os.path.basename(url)

            count += 1

        else:
            continue

        if image_dl is not None:
            yield (image_dl, file_name, caption)
            
    print(f"{count} images downloaded")


q, k, v = zip(*get_moondream_data(1000))


def save_images(file_generator):
    for image_file in file_generator:
        try:
            image_buffer = io.BytesIO(image_file.read())
            image = pillow_image.open(image_buffer)
            
            image_path = os.path.join("images", image_file.name)
            image.save(image_path)
            print(f"{image_file.name} saved successfully.")
            
        except Exception as e:
            print(f"Error: {e}")
            
        finally:
            image_file.close()


qx = [save_images(file_gen) for file_gen in q]


csv_path = "moondream_2.csv"

def moondream_csv(path: List, desc: List):
    print("Writing to csv..")

    md_dict = {"image_path": path, "caption": desc}

    moondream_df = pd.DataFrame(md_dict)

    moondream_df.to_csv(csv_path, index=False)

    print("Csv transfer complete...")


moondream_csv(k, v)

print("Moondream to Kaggle porting complete")
