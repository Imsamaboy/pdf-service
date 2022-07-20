# PDF -> Docx

### Общие сведения

Задача была реализована на языке **Python** (3.10) с использованием **OpenCv, Flask, Tesseract, python-docx**    и других вспомогательных библиотек.

Сервис может распознать данные в pdf файле и перевести их в новый созданный docx файл.

### How to use?
По данному url ```http://localhost:5000/convert-pdf``` можно отправить GET http-запрос с (binary) body в виде исходного pdf файла.

Сервис обработает данный файл и вернёт docx файл (также, он будет сохранён в корне директории самого проекта).

### TODO
Однако есть недочёты в виде распознавания рукописного и смешанного (латиница и кириллица, например) текста.

Данная проблема решается, например, обучением нейронной сети по распознаванию рукописного текста и дальнейшей интеграции в сервис.
Что касается нескольких типов алфавитов, то можно использовать другой движок OCR или другую адаптацию Tesseract для Python.

### Запуск сервис
**Docker**
```
docker build -t pdf-service-image .
docker run pdf-service-image
```

**CMD** (нужно иметь Python 3.10 / создать venv)

В корне проекта прописать:
```
pip install -r requirements.txt
cd /main
python app.py
```