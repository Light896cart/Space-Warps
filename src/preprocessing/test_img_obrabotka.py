from PIL import Image
import numpy as np

# Открытие изображения
image = Image.open('downloaded_images/page_11_block_10_frame_1.jpg')

# Преобразование в массив NumPy
image_array = np.array(image)

# Вывод формы
print(image_array.shape)