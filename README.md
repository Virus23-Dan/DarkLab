# DarkLab AI

**DarkLab** est une plateforme d'intelligence artificielle conçue pour répondre à une variété de questions dans différents domaines, gérer des événements calendaires, effectuer des recherches via l'API Groq, et entraîner un modèle de Deep Learning personnalisé. Son interface web conviviale permet une interaction fluide avec l'utilisateur, tout en intégrant des outils spécialisés selon le domaine de la question.

---

## Fonctionnalités principales

- **Assistance en plusieurs domaines** : mathématiques, sciences, programmation, médecine, finance, histoire, littérature, etc.
- **Détection automatique du domaine** et adaptation des prompts.
- **Gestion de calendrier** : ajout, visualisation d’événements.
- **Recherche via API Groq**.
- **Interface Web interactive** : chat, indicateurs de statistiques, configuration utilisateur.
- **Entraînement et sauvegarde d’un modèle neuronal personnalisé**.
- **Utilisation de GPT-4 / GPT-3.5-turbo** pour générer des réponses contextuelles et structurées.

---

## Structure du projet

- **Backend API** : Flask, héberge l'interface web et gère les requêtes utilisateur.
- **Interface Web** : HTML, CSS, JavaScript pour l’interaction utilisateur.
- **Modules ML** : modèles neuronaux avec PyTorch, gestion des entraînements et sauvegarde.
- **Outils supplémentaires** : gestion du calendrier, API Groq, gestion de contexte.

---

## Prérequis

- Python 3.8 ou supérieur
- Packages Python :
  - Flask
  - openai
  - torch
  - numpy
  - requests
  - argparse

- Clés API nécessaires :
  - **DeepAI API Key** : pour utiliser GPT-4 ou GPT-3.5
  - **API Groq** : pour la recherche (optionnel, modifiable dans le code)

## Installation

Clone le repository :

```bash
git clone https://github.com/Virus23-Dan/DarkLab.git
cd darklab-ai
```

Installation des dépendances :
```
pip install -r requirements.txt
```

