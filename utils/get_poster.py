import os
from dotenv import load_dotenv
import requests
import numpy as np
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
import base64

from logging_custom.logger import Logger

load_dotenv()
POSTER_DIR = os.getenv("POSTER_DIR")

logger = Logger("GetPoster").get_logger()
headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }

def save_poster_directory(img, movieId):
    # save image in poster directory
    if img and movieId:
        img_filename = str(movieId) + ".png"
        save_path = os.path.join(POSTER_DIR, img_filename)
        img.save(save_path)
        logger.info(f"Image saved at {save_path}")
        return True
    else:
        return False

def search_image_directory(movieId):
    # search image in poster directory
    img_name = str(movieId)+".png"
    if img_name in os.listdir(POSTER_DIR):
        logger.info(f"Movie {movieId} image found in directory")
        img = Image.open(os.path.join(POSTER_DIR, img_name))
        return img
    else:
        return None

def get_image(imdb_id, movieId, links_helper):
    try:
        search_img = search_image_directory(movieId)
        if search_img:
            return search_img

        if links_helper.search_img_data(movieId):
            logger.info(f"Image found for {movieId} in database. Skipping download.")
            base64_str = links_helper.search_img_data(movieId)
            image_bytes = BytesIO(base64.b64decode(base64_str))
            img = Image.open(image_bytes)
            if not save_poster_directory(img, movieId):
                logger.error(f"Failed to save image for movie {movieId}")
            return img
        # if image not in links_helper, fetch from web
        logger.info(f"Fetching image for IMDb ID {imdb_id}")
        # get image url
        soup = BeautifulSoup(requests.get(f"https://www.imdb.com/title/{imdb_id}", headers=headers).content, 'html.parser')
        # img_url = soup.find('img', attrs={'class': "poster w-full"})['src']
        img_url = soup.find('img', attrs={'class': 'ipc-image'})['src']

        # get image
        response = requests.get(img_url, headers=headers)
        # convert image to base64
        # base64_str = base64.b64encode(response.content).decode('utf-8')
        # # update links_helper
        # links_helper.update_links_img_data(movieId, base64_str)
        
        # read image in a numpy array
        image_bytes = BytesIO(response.content)
        img = Image.open(image_bytes)
        # save image in poster directory
        if not save_poster_directory(img, movieId):
            logger.error(f"Failed to save image for movie {movieId}")

        return img
        
    except Exception as e:
        logger.error(f"Error fetching image for IMDb ID {imdb_id}: {e}")
        return None
    
def get_description(imdb_id):
    try:
        logger.info(f"Fetching description for IMDb ID {imdb_id}")
        soup = BeautifulSoup(requests.get(f"https://www.imdb.com/title/{imdb_id}", headers=headers).content, 'html.parser')
        description = soup.find('span', attrs={'data-testid': 'plot-l'}).text
        return description
    except Exception as e:
        logger.error(f"Error fetching description for IMDb ID {imdb_id}: {e}")
        return "Description not available."

def get_random_image(location='H:/pou'):
    try:
        # logger.info(f"Getting Random images from {location}")
        images = os.listdir(location)
        images = [img for img in images if img.endswith((".jpg", ".png", ".jpeg", ".webp"))]
        img = os.path.join(location, images[np.random.randint(0, len(images))])
        return img
    except Exception as e:
        raise e

        