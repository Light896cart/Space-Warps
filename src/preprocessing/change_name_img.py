import os

# Папка с изображениями
download_folder = "downloaded_images"

# Перебираем все файлы в папке
for filename in os.listdir(download_folder):
    if filename.startswith("block_") and "page_" not in filename:
        # Разделяем имя файла на части
        name_parts = filename.split("_")

        # Ожидаем формат: block_X_frame_Y.jpg
        if len(name_parts) >= 4 and name_parts[0] == "block" and name_parts[2] == "frame":
            # Формируем новое имя
            new_name = f"page_1_{filename}"
            old_path = os.path.join(download_folder, filename)
            new_path = os.path.join(download_folder, new_name)

            # Переименовываем файл
            os.rename(old_path, new_path)
            print(f"Переименован: {filename} → {new_name}")
        else:
            print(f"Пропущен файл (неправильный формат): {filename}")