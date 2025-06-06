# Документация по сервису кластеризации

## Описание

Этот проект представляет собой **сервис кластеризации**, который выполняет извлечение и классификацию сущностей из текстовых данных. Он использует предобученную модель для классификации сущностей в тексте (NER), а также модель для извлечения ключевых данных, таких как название компании, должность, объем данных и другие важные атрибуты. Сервис принимает текстовые файлы, извлекает сущности и сохраняет результаты в базе данных.

Процесс обработки включает:
1. Загрузка текстового файла из облачного хранилища (S3).
2. Извлечение ключевых сущностей с использованием модели Named Entity Recognition (NER).
3. Кластеризация сущностей в разные группы (например, компании, должности, телефоны).
4. Сохранение извлеченных данных в базу данных.
5. Обработка и формирование финального отчета с выделением сущностей и их значений.

## Функции

- **Загрузка текстового файла**: Текстовый файл загружается из облачного хранилища S3.
- **Извлечение сущностей**: Извлечение сущностей из текста с использованием модели NER (Named Entity Recognition).
- **Кластеризация**: Классификация сущностей и их кластеризация по категориям (например, компания, должность, телефон).
- **Обновление базы данных**: Извлеченные данные сохраняются в базе данных для дальнейшей обработки.

## Установка

### Требования

- Python 3.8+
- Библиотеки:
  - `torch` (для работы с моделью NER)
  - `requests` (для отправки запросов)
  - `boto3` (для работы с S3)
  - `psycopg2` (для работы с базой данных)
  - `transformers` (для работы с моделью NER)
  - `sklearn` (для кластеризации)
  - `dotenv` (для работы с переменными окружения)

### Шаги установки

1. Клонируйте репозиторий:
   ```bash
   git clone <url-репозитория>
   cd <папка-проекта>


Как работает кластеризация
1. Загрузка файла из S3
Текстовый файл загружается из облачного хранилища S3 в локальную временную папку.

python
Копировать
file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
temp_file_path = os.path.join(temp_dir, file_key)

with open(temp_file_path, 'wb') as temp_file:
    temp_file.write(file_obj['Body'].read())
2. Извлечение сущностей с помощью модели NER
С помощью модели NER извлекаются ключевые сущности из текста, такие как имена, компании, должности и телефоны.

python
Копировать
def parse_result(results):
    readable_result = []
    for pred in results:
        label = id_to_label[int(pred["entity"].split("_")[1])]
        readable_result.append({
            "word": pred["word"],
            "entity": label,
            "score": pred["score"],
            "start": pred["start"],
            "end": pred["end"]
        })
    ...
3. Кластеризация сущностей
Извлеченные сущности группируются и классифицируются с использованием алгоритма кластеризации (например, KMeans).

python
Копировать
def get_clusters(text):
    results = ner_pipeline(text)
    return parse_result(results)
4. Обработка и сохранение данных
После кластеризации данные обрабатываются, извлекаются важные атрибуты, такие как название компании, телефон, должность, и сохраняются в базу данных.

python
Копировать
def update_record(record_id, company_name, position, description, data_type, amount, person_name, next_place, telephone):
    ...
5. Удаление временных файлов
После завершения обработки все временные файлы удаляются для очистки пространства.

python
Копировать
os.remove(temp_file_path)
Лицензия
Этот проект лицензирован под лицензией MIT — смотрите файл LICENSE для подробностей.