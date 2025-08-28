import pandas as pd
import os
from astropy.io import fits
import numpy as np
from PIL import Image
from matplotlib.colors import AsinhNorm

from src.preprocessing.get_image_ps1 import get_im


def download_images_from_csv(csv_path, output_dir, size=75, filters="g", im_format="jpg", save_as="jpg"):
    """
    Загружает изображения из PS1 по координатам.

    csv_path: путь к CSV с 'id', 'ra', 'dec'
    output_dir: куда сохранять
    size: размер в угловых секундах
    filters: фильтр, например "i"
    im_format: формат, в котором запрашивать у сервера (лучше "fits", чтобы получить данные)
    save_as: в каком формате сохранить локально (например, "jpg", "png")
    """
    df = pd.read_csv(csv_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    downloaded_count = 0
    filter_list = [filters] if isinstance(filters, str) else filters

    for _, row in df.iterrows():
        obj_id = str(row['id'])
        ra = float(row['ra'])
        dec = float(row['dec'])

        fpath = os.path.join(output_dir, f"{obj_id}.{save_as}")

        if os.path.exists(fpath):
            downloaded_count += 1
            continue

        try:
            # Загружаем данные в формате FITS (даже если хотим сохранить как JPG)
            img = get_im(ra, dec, size=size, filters=filter_list, im_format="fits", color=False)

            if img is None:
                print(f"get_im вернул None для {obj_id}")
                continue

            # Извлекаем данные из FITS
            if isinstance(img, fits.HDUList):
                data = img[0].data  # обычно данные в PRIMARY HDU
                if data is None or data.size == 0:
                    print(f"Пустые данные FITS для {obj_id}")
                    continue
            elif isinstance(img, np.ndarray):
                data = img
            else:
                print(f"Неожиданный тип данных: {type(img)}")
                continue

            # Обрабатываем данные: инвертируем, нормализуем
            # Убираем NaN
            data = np.nan_to_num(data, nan=0.0)

            # Инвертируем (звезды яркие, фон темный)
            # Можно не инвертировать — зависит от предпочтений
            # data = np.max(data) - data  # инверсия

            # Нормализуем в диапазон 0-255 с помощью asinh (как в Aladin)
            norm = AsinhNorm()  # хороший выбор для астрономических изображений
            image_scaled = norm(data)
            image_scaled = (image_scaled * 255).astype(np.uint8)

            # Конвертируем в PIL Image
            pil_image = Image.fromarray(image_scaled)

            # Сохраняем как JPG или PNG
            pil_image.save(fpath, quality=95)

            downloaded_count += 1
            if downloaded_count % 10 == 0:
                print(f"Загружено {downloaded_count} изображений...")

        except Exception as e:
            print(f"Ошибка при обработке объекта {obj_id}: {e}")

    print(f"Загрузка завершена. Всего сохранено: {downloaded_count} изображений.")

for c in range(1):
    # Форматируем номер так, чтобы был с ведущими нулями: 0002, 0003, ..., 0019
    chunk_str = f"{c:04d}"  # 2 → '0002', 15 → '0015'

    csv_path = rf'D:\Code\Space_Warps\data\reg\balanced_2001_by_class_cycle.csv'
    output_dir = rf'D:\Code\Space_Warps\data\image_data\img_csv'

    # Пример вызова:
    download_images_from_csv(
        csv_path=csv_path,
        output_dir=output_dir,
        size=75,
        filters="i",
        im_format="fits",
        save_as="jpg"
    )