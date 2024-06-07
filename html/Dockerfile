# Utilisez une image de base Python
FROM python:3.8-slim

# Définissez le répertoire de travail
WORKDIR /app

# Copiez les fichiers de dépendances et installez les paquets nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiez votre code source dans l'image
COPY src/ .

# Commande pour exécuter votre application
CMD ["python", "app.py"]
