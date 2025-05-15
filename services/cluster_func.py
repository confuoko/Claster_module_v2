import os

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification,  pipeline
import psycopg2

id_to_label = {0: 'DOC', 1: 'MDT', 2: 'NAME', 3: 'O', 4: 'ORG', 5: 'POS', 6: 'TEL', 7: 'VOL'}

# Загружаем обученную модель и токенизатор из папки ner-model-1.0
model_path = "services/ner-model-2.0"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Создаём пайплайн NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=device)


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

    current_group = None
    result = {}
    current_phrase = ''

    def add_pair(key, value):
        if key not in result:
            result[key] = []
        result[key].append(value)

    for res in readable_result:
        word = res['word']
        group = res['entity']
        if group == "O":
            if current_group != None:
                add_pair(current_group, current_phrase)
            current_group = None
            current_phrase = ''
            continue
        if word.startswith('##'):
            current_phrase = current_phrase + word.replace('##', '')
        elif group == current_group:
            current_phrase = current_phrase + ' ' + word
        else:
            if current_group != None:
                add_pair(current_group, current_phrase)
            current_group = group
            current_phrase = word

    for key in result:
        result[key] = list(set(result[key]))

    return result

def get_clusters(text):

    results = ner_pipeline(text)
    print(100)
    return parse_result(results)


def update_record(record_id, company_name, position, description, data_type, amount, person_name, next_place, telephone):
    # Загрузка переменных из .env
    load_dotenv()

    # Получаем connection string
    connection_string = os.getenv('DATABASE_URL')

    try:
        # Подключение к БД
        connection = psycopg2.connect(connection_string)
        print("Успешное подключение к БД")
        cursor = connection.cursor()

        # Обновление нескольких полей
        update_query = """
            UPDATE myusers_callitem
            SET company_name = %s,
                position = %s,
                description = %s,
                data_type = %s,
                amount = %s,
                person_name = %s,
                next_place = %s,
                telephone = %s
            WHERE id = %s;
        """
        cursor.execute(update_query, (company_name, position, description, data_type, amount, person_name, next_place, telephone, record_id))
        connection.commit()

        print(f"Запись с id={record_id} успешно обновлена.")

    except Exception as e:
        print(f"Ошибка при обновлении записи: {e}")

    finally:
        # Закрытие соединения
        if connection:
            cursor.close()
            connection.close()