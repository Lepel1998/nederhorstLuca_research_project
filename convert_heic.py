# convert HEIC image to other compatible format
import pyheif
from PIL import Image

def ConvertImage(photo_path):

    heif_photo = pyheif.read(photo_path)
    
    photo = Image.frombytes(
        heif_photo.mode,
        heif_photo.size,
        heif_photo.data,
        "raw",
        heif_photo.mode,
        heif_photo.stride,
    )

    jpeg_photo_path = 'converted_photo.jpg'
    return jpeg_photo_path