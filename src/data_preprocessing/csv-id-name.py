import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path):
    """
    Загружает и отображает изображение по указанному пути с помощью matplotlib.
    """
    img = mpimg.imread(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')  # Отключаем оси
    plt.show()

show_image('data/image_data/img_csv_0001/12326474500150005914600060103.jpg')