FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY credentials.json /app/credentials.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json

# Installer les dépendances système pas sur de garder l.14
# COPY app.py /app/app.py              

COPY index.html /app/index.html
COPY Rubie_portrait.png /app/Rubie_portrait.png
COPY mosaic.jpg /app/mosaic.jpg


# Créer le dossier flags s'il n'existe pas déjà dans l'image
RUN mkdir -p /app/flags
COPY flags/*.png /app/flags/

# add the pdf files
RUN mkdir -p /app/douments
COPY documents/*.pdf /app/documents/
# add embedding model files
RUN mkdir -p /app/paraphrase-multilingual-mpnet-base-v2
COPY paraphrase-multilingual-mpnet-base-v2/  /app/paraphrase-multilingual-mpnet-base-v2/

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_CLOUD_PROJECT=btc-chatbot-pro
ENV VERTEX_LOCATION=us-central1


# Exposer le port
EXPOSE 8080
# Remplacez la dernière ligne dans votre Dockerfile

CMD ["python", "app.py"]

