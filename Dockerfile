FROM python:3.8

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "scripts/app.py"]
