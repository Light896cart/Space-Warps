from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import os
import requests

# Папка для сохранения изображений
download_folder = "downloaded_images"
os.makedirs(download_folder, exist_ok=True)

# Настройки браузера
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
# options.add_argument("--headless")  # Опционально: запуск без открытия окна

driver = webdriver.Chrome(options=options)

# Цикл по страницам
for page_num in range(2, 23):  # от 2 до 22 включительно
    url = f" https://www.zooniverse.org/projects/aprajita/space-warps-esa-euclid/collections/paola/lenses?page={page_num}"
    print(f"\nОткрываю страницу {page_num}...")

    try:
        driver.get(url)

        # Ждём загрузки блоков
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "collection-subject-viewer"))
        )
        time.sleep(2)  # Дополнительное время на прогрузку

        # Получаем все блоки
        blocks = driver.find_elements(By.CLASS_NAME, "collection-subject-viewer")
        print(f"Найдено блоков на странице {page_num}: {len(blocks)}")

        for block_index, block in enumerate(blocks):
            print(f"  Обработка блока {block_index + 1}/{len(blocks)} на странице {page_num}...")

            # Прокручиваем до блока
            driver.execute_script("arguments[0].scrollIntoView();", block)
            time.sleep(1)

            # Скачиваем первое изображение
            try:
                img = block.find_element(By.CLASS_NAME, "subject.pan-active")
                img_url = img.get_attribute("src")
                print("    Скачиваем изображение:", img_url)
                response = requests.get(img_url)
                with open(os.path.join(download_folder, f"page_{page_num}_block_{block_index + 1}_frame_1.jpg"), "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"    Не удалось найти первое изображение в блоке {block_index + 1}: {e}")
                continue

            # Нажимаем на кадры 2, 3, 4 через JS
            for i in range(2, 5):
                try:
                    pip_input = block.find_element(By.XPATH, f".//label[contains(@class, 'subject-frame-pip')][{i}]//input")
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", pip_input)
                    driver.execute_script("arguments[0].click();", pip_input)
                    time.sleep(1.5)

                    img = WebDriverWait(block, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "subject.pan-active"))
                    )
                    img_url = img.get_attribute("src")
                    print(f"    Скачиваем кадр {i}:", img_url)
                    response = requests.get(img_url)
                    with open(os.path.join(download_folder, f"page_{page_num}_block_{block_index + 1}_frame_{i}.jpg"), "wb") as f:
                        f.write(response.content)

                except Exception as e:
                    print(f"    Не удалось найти кадр {i} в блоке {block_index + 1}: {e}")
                    continue

    except Exception as e:
        print(f"Ошибка при обработке страницы {page_num}: {e}")
        continue

# Закрываем браузер
driver.quit()
print("Парсинг завершён.")