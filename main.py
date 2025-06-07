import os
import re

import torch
import runpod
from dotenv import load_dotenv
import boto3

from services.cluster_func import get_clusters, update_record


def handler(event):
    """
        Принимает id записи в БД и имя файла в хранилище s3
        Берет из хранилища s3 файл, сохраняет его в папку temp.
        Берет текст из файла и выделяет ключевые сущности.
        Сохраняет эти сущности в Postgres
    """

    torch_variable = str(torch.cuda.is_available())
    if torch.cuda.is_available():
        print("✅ CUDA доступна (используется GPU)")
    else:
        print("❌ CUDA недоступна (используется CPU)")

    load_dotenv()

    # 1. Скачиваем текстовый файл из хранилища s3 в папку temp
    file_key = event['input']['file_key']
    item_id = event['input']['item_id']

    #file_key = 'разговор8транс.txt'
    #item_id = 27

    print(f"Получен запрос на обработку файла: {file_key}")
    print(f"ID элемента: {item_id}")

    # Параметры хранилища s3
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name = "whisper-audiotest"

    s3 = boto3.client(
        's3',
        endpoint_url="http://storage.yandexcloud.net",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file_key)

    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(file_obj['Body'].read())
    print(f'📁 Файл сохранён во временную папку: {temp_file_path}')

    # === Считываем текст в переменную ===
    with open(temp_file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    print(file_content)

    # === Обрабатываем текст через кластеризатор
    result = get_clusters(file_content)

    # Все возможные ключи
    id_to_label = {
        0: 'DOC', # тип документа, Purpose.data_type, модель ожидает строку
        1: 'MDT', # время следующей встречи, модель ожидает строку
        2: 'NAME', # имя представителя компании, Opponent.name модель ожидает строку
        3: 'O',
        4: 'ORG',  # название компании, склеивем все строки в одну строку
        5: 'POS',  # Должность
        6: 'TEL',  # телефон, берем первое значение из списка
        7: 'VOL',  # объем данных, Purpose.amount модель ожидает число
    }

    # Добавляем отсутствующие ключи с None
    full_result = {
        label: result.get(label) if label in result else None
        for label in id_to_label.values()
    }
    print(full_result)
    print("Начинаем обработку полученных значений")

    # Обработка значений
    # 1. Название компании-контрагента company_name
    print("1. ORG: Название компании-контрагента company_name ")
    company_name = None
    org_value = full_result.get('ORG', None)
    if org_value:
        company_name = ' '.join(org_value)  # Склеиваем все строки в одну строку
    print(f"company_name: {org_value}")

    # 2. Телефон компании company_phone
    print("2. TEL: Телефон компании company_phone")
    tel_value = full_result.get('TEL', None)
    if tel_value:
        company_phone = tel_value[0]  # Берем первый телефон из списка
    else:
        company_phone = None
    print(f"company_phone: {company_phone}")


    # 3. Тип документа data_type
    print("3. DOC: Тип документа data_type")
    doc_value = full_result.get('DOC', None)
    if doc_value:
        doc_type = ' '.join(doc_value)  # Склеиваем все строки в одну строку
    else:
        doc_type = "Тип не определен"
    print(f"data_type: {doc_type}")

    # 4. Предполагаемый объем документов data_value, description, word
    print("4. VOL: Предполагаемый объем документов")
    vol_value = full_result.get('VOL', None)
    if vol_value:
        # Извлекаем только числа из строки
        numbers = re.findall(r'\d+', vol_value[0])
        data_value = int(numbers[0]) if numbers else 0  # Берем первое найденное число
        description = ", ".join(vol_value) if vol_value else "Не распознано"
        print(f"4.1 description: полное описание данных из VOL - {description}")

        # Извлекаем слово после числа

        # Извлекаем все слова (состоящие только из букв)
        words = re.findall(r'\b[а-яА-Яa-zA-Z]+\b', description)
        word = ", ".join(words) if words else "Не распознано"
        print(f"4.2 word: только слова из VOL - {word}")
        print(f"4.3 amount: первое число из VOL - {data_value}")

    else:
        data_value = 0
        word = 'Тип не определен'
        description = 'Запрос не определен'

    # 5. Должность position_value
    print("5. POS: Должность position_value")
    position_val = full_result.get('POS', None)
    if position_val:
        position_value = position_val[0]
    else:
        position_value = None
    print(f"position_value - {position_value}")

    # 7 Следующая встреча
    print("7. MDT: Следующая встреча")
    place_val = full_result.get('MDT', None)
    if place_val:
        place_value = place_val[0]
    else:
        place_value = None
    print(f"Следующая встреча - {place_value}")


    # 8 Имя представителя
    print("7. NAME: Имя представителя")
    person_name_val = full_result.get('NAME', None)
    if person_name_val:
        person_name = ", ".join(person_name_val) if person_name_val else "Не распознано"
    else:
        person_name = None
    print(f"Имя представителя - {person_name}")

    print("----------------------------------------------")
    print("Подготовлены следующие данные для записи в БД:")
    print(f"ID объекта: {item_id}")
    print(f"Название компании: {company_name}")
    print(f"Должность: {position_value}")
    print(f"Описание объема: {description}")
    print(f"Тип документа: {doc_type}")
    print(f"Периодичность/дата: {data_value}")
    print(f"Контактные лица: {person_name}")
    print(f"Место действия: {place_value}")
    print(f"Телефон компании: {company_phone}")


    # пушим данные в БД
    update_record(item_id, company_name, position_value, description, doc_type, data_value, person_name, place_value,
                  company_phone)

    # === Удаляем временные файлы ===
    os.remove(temp_file_path)
    print("🧹 Временные файлы удалены.")

if __name__ == "__main__":
    runpod.serverless.start({'handler': handler})
    #handler(None)