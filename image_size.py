import PIL
from PIL import Image

folder_path = './Data/Grayscale_2/' 
img = '1664.jpg'
img = folder_path + img

width, height = Image.open(img).size
print(width, height)
