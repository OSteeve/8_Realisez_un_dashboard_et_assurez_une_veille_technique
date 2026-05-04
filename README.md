# 8_Realisez_un_dashboard_et_assurez_une_veille_technique
amélioration du P7(Implémentez un modèle de scoring) amélioration du dashbord et P6 (classifiication des biens de conso) pour la partie veille technologique, utilisation d'un modèle moderne Transformer..  
Maintenant nous souhaitons utiliser l’API pour réaliser un dashboard à destination de nos chargés de
relation client. Ils le réclament depuis longtemps afin de pouvoir mieux expliquer à leurs clients les
décisions, et parfois les revoir si nécessaire.


Oui, pour un projet de soutenance/portfolio, tu peux faire beaucoup plus efficace. Voici une version plus concise et plus “pro” :

---

# P8 – Dashboard interactif de scoring client & veille technique

## Contexte

Dans le projet **P7**, une API de scoring client a été développée afin de prédire l’accord ou le refus d’un dossier de prêt.

L’objectif de ce projet est de développer une interface métier interactive permettant aux chargés de relation client de mieux comprendre, expliquer et éventuellement réévaluer les décisions du modèle.

Le dashboard est développé avec Streamlit et s’appuie sur l’API de prédiction développée dans le projet **P7** avec FastAPI.

---

## Objectifs

Le dashboard permet de :

* visualiser la décision du modèle (**accord/refus**) ;
* afficher la probabilité associée et la distance au seuil de décision ;
* interpréter le score de manière compréhensible pour un utilisateur non technique ;
* consulter les principales informations descriptives d’un client ;
* comparer un client à l’ensemble du portefeuille ou à des profils similaires ;
* filtrer dynamiquement les comparaisons selon différentes variables ;
* proposer des visualisations respectant des critères d’accessibilité (WCAG) ;
* déployer l’application dans le Cloud pour un accès distant.

### Fonctionnalités expérimentales

* simulation de modification des variables client ;
* recalcul dynamique du score ;
* création d’un nouveau dossier client.

---

## Veille technique

Une seconde partie du projet consiste à réaliser une veille technologique autour des architectures Transformer.

Cette étude reprend le notebook de classification d’images du projet P6 basé sur :

VGG16

Objectif :

* comparer les performances des architectures CNN classiques avec les Transformers ;
* analyser les avantages, limites et coûts de calcul.

---

## Technologies

* Python
* Streamlit
* FastAPI
* Pandas
* Scikit-learn
* SHAP
* Plotly
* GitHub Actions
* Amazon EC2

---

## Installation

```bash
git clone <repo_url>
cd P8
pip install -r requirements.txt
```

---

## Lancement

```bash
streamlit run app/streamlit_app.py
```

⚠️ L’API du projet **P7** doit être accessible pour permettre les prédictions.

---

## Réutilisation de projets précédents

Ce projet réutilise certains notebooks, modèles et données issus des projets **P6** et **P7**.

Ces fichiers ont été copiés localement afin de rendre le projet autonome. Les mises à jour de P6 ou P7 ne sont pas synchronisées automatiquement.

---

Ça ressemble davantage à un README qu’un recruteur, un mentor ou un jury va réellement lire jusqu’au bout.
