# Prédiction de Sentiment pour Air Paradis

Chez **Marketing Intelligence Consulting (MIC)**, nous avons récemment été mandatés par Air Paradis pour développer un prototype d’IA capable de prédire le sentiment associé à un tweet. Cette initiative vise à anticiper et gérer les bad buzz sur les réseaux sociaux, une problématique cruciale pour la compagnie aérienne.

## Présentation des Trois Approches

Pour répondre au besoin d'Air Paradis, nous avons exploré trois approches différentes pour la prédiction de sentiment :

- **Modèle sur mesure simple** : Régression logistique avec des caractéristiques TF-IDF.
- **Modèle sur mesure avancé** : Modèles basés sur des réseaux de neurones profonds avec différentes techniques de plongement de mots (embeddings).
- **Modèle avancé BERT** : Utilisation du modèle pré-entraîné BERT pour la prédiction de sentiment.

### 1. Modèle sur Mesure Simple

**Méthodologie :**

- **TF-IDF** : Transformation des tweets en représentations vectorielles basées sur la fréquence des mots.
- **Régression Logistique** : Utilisation de la régression logistique pour la classification binaire.

**Résultats :**

- Accuracy : environ 0.773 à 0.775
- Temps d’entraînement : Quelques minutes

Nous avons également testé des techniques de prétraitement du texte telles que la lemmatisation et la stemmatisation :

- **Lemmatisation** : Accuracy de 0.775
- **Stemmatisation** : Accuracy de 0.773

### 2. Modèle sur Mesure Avancé

**Méthodologie :**

- **Embeddings** : Utilisation d'embeddings pour capturer le contexte des mots.
- **CNN et LSTM** : Modèles de réseaux de neurones pour traiter les séquences de texte.

**Résultats :**

- **Embedding CNN** : Accuracy de 0.854 (meilleur score obtenu)
- **Embedding LSTM** : Accuracy de 0.799
- Temps d’entraînement : Quelques heures (2.8 heures pour CNN, 9.8 heures pour LSTM)

### 3. Modèle Avancé BERT

**Méthodologie :**

Nous avons utilisé un Mac Mini pour entraîner le modèle BERT, en utilisant le modèle pré-entraîné TinyBERT avec des données préparées et tokenisées, puis en l'entraînant sur plusieurs époques avec une optimisation adaptée aux performances du Mac Mini.

**Résultats :**

- Accuracy : 0.768
- Temps d’entraînement : Environ 7 heures

## Synthèse des Résultats

| Modèle                                 | Accuracy | Temps d'entraînement |
|----------------------------------------|----------|----------------------|
| TF-IDF Régression Logistique (Stemming)| 0.773    | < 10 min             |
| TF-IDF Régression Logistique (Lemmatisation) | 0.775 | < 10 min             |
| **Embedding CNN**                      | **0.854**| ~2.8 heures          |
| Embedding LSTM                         | 0.799    | ~9.8 heures          |
| BERT                                   | 0.768    | ~7 heures            |

## Démarche MLOps

Pour assurer la qualité et la reproductibilité de nos expérimentations, nous avons mis en place une démarche MLOps complète. Voici les étapes clés que nous avons suivies :

1. **Traçabilité des Expérimentations avec MLFlow** :
    - **Enregistrement des Expérimentations** : Chaque essai de modèle, y compris les hyperparamètres et les résultats, a été enregistré dans MLFlow.
    - **Comparaison des Modèles** : Les visualisations de MLFlow ont été utilisées pour comparer les performances des différents modèles.

2. **Centralisation du Stockage des Modèles** :
    - Tous les modèles entraînés ont été stockés de manière centralisée dans MLFlow.

3. **Reproductibilité** :
    - **Pipelines Reproductibles** : Nous avons utilisé des pipelines automatisés pour assurer que les expérimentations puissent être reproduites à tout moment.

## Déploiement sur le Cloud

Le modèle CNN, étant le plus performant, a été déployé via une API sur une plateforme Cloud gratuite (Azure). Nous avons intégré un pipeline de déploiement continu avec des tests unitaires pour garantir la robustesse de l'API.

## deploiement azure: 
az container create \
    --resource-group myResourceGroup \
    --name mycontainer \
    --image emiled/opc \
    --cpu 2 \
    --memory 4 \
    --port 5005 \
    --dns-name-label mycontainerdns4 \
    --environment-variables WEBSITES_PORT=5005

## Conclusion

Notre projet pour Air Paradis démontre l'efficacité des modèles avancés, particulièrement des réseaux de neurones profonds avec embeddings. Le modèle Embedding CNN a montré des performances exceptionnelles, surpassant même le modèle BERT. En intégrant des pratiques MLOps, nous avons assuré une gestion optimisée et reproductible de nos expérimentations, offrant ainsi une solution robuste et prête pour une mise en production.

Pour toute question ou demande de projet similaire, n'hésitez pas à nous contacter chez **Marketing Intelligence Consulting**. Nous sommes ravis d'apporter notre expertise en IA et MLOps pour répondre à vos besoins en marketing digital.

Concrètement, il est nécessaire de simplifier les aides, d'ajouter des contributions volontaires de personnes souhaitant donner sans être imposées, et surtout de cesser d'utiliser les aides et les taxes à des fins politiques. Tout comme l'hygiène au 19ème et au début du 20ème siècle, qui s'est répandue grâce à des initiatives indépendantes avant d'être adoptée par la politique, nous devons aujourd'hui agir indépendamment pour rendre les aides accessibles et efficaces pour tous.



