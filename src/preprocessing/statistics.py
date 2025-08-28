import pandas as pd
rqqew = r'D:\Code\Space_Warps\data\reg\balanced_2001_by_class_cycle — копия.csv'
# Загружаем CSV
# Настройка: отключаем ограничение на отображение данных
pd.set_option('display.max_rows', None)  # показывать все строки
pd.set_option('display.max_columns', None)  # все колонки
pd.set_option('display.width', None)  # ширина вывода
pd.set_option('display.max_colwidth', None)
df = pd.read_csv(rqqew)  # замени на путь к твоему файлу

# 1. Общая информация
print("=== ОБЩАЯ ИНФОРМАЦИЯ О ДАННЫХ ===")
print(f"Размер таблицы: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("\nИмена столбцов:")
print(df.columns.tolist())

print("\nТипы данных:")
print(df.dtypes)

# 2. Сколько уникальных значений в столбце 'class'?
if 'class' in df.columns:
    print(f"\n=== СТАТИСТИКА ПО СТОЛБЦУ 'class' ===")
    unique_count = df['subclass'].nunique()
    print(f"Количество уникальных значений в 'class': {unique_count}")

    print(f"\nЧастота каждого значения в 'class':")
    print(df['subclass'].value_counts().sort_index())
else:
    print("\nСтолбец 'class' не найден. Доступные столбцы:")
    print(df.columns.tolist())

# 3. Пропущенные значения
print(f"\n=== ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ ===")
print(df.isnull().sum())

# 4. Описательная статистика по числовым столбцам
print(f"\n=== ОПИСАТЕЛЬНАЯ СТАТИСТИКА (для числовых столбцов) ===")
print(df.describe())