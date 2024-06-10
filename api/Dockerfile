# Utiliser l'image de base python:3.9-slim
FROM python:3.9-slim

# Installer wget et bzip2 nécessaires pour installer Miniconda
RUN apt-get update && apt-get install -y wget bzip2 && \
    rm -rf /var/lib/apt/lists/*

# Télécharger et installer Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Mettre à jour le PATH pour utiliser conda
ENV PATH /opt/conda/bin:$PATH

# Copier le fichier YAML pour créer l'environnement conda
COPY bert_sentiment_analysis_x86.yaml /tmp/bert_sentiment_analysis_x86.yaml

# Créer l'environnement conda à partir du fichier YAML
RUN conda env create -f /tmp/bert_sentiment_analysis_x86.yaml

# Assurer que conda est initialisé dans le shell
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate bert_sentiment_analysis" >> ~/.bashrc

# Utiliser le shell bash avec conda initialisé
SHELL ["/bin/bash", "--login", "-c"]

# Activer l'environnement conda et installer les dépendances de l'application
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate bert_sentiment_analysis

# Copier les fichiers de l'application
WORKDIR /api
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py main.py
COPY cnn_model.keras cnn_model.keras
COPY cnn_model_tokenizer.json cnn_model_tokenizer.json

# Copier le script entrypoint dans le conteneur et le définir comme point d'entrée
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
