import os
import json
import time
import asyncio
import socket
from datetime import datetime
import openai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import requests
import argparse
from flask import Flask, request, jsonify, render_template_string

# =========================
# Configuration et utilitaires
# =========================

class Config:
    def __init__(self):
        self.filename = 'config.json'
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 20,
                'max_tokens': 10000000000,
                'temperature': 0.7,
                'openai_api_key': '',  # Charger depuis env ou fichier
            }
            self.save(data)
        self.__dict__.update(data)

    def save(self, data):
        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=4)

    def is_valid(self):
        return bool(self.openai_api_key)

config = Config()

# Charger la clé API DeepAI une seule fois
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', config.openai_api_key)

# =========================
# Mémoire contextuelle
# =========================

class ContextualMemory:
    def __init__(self, max_contexts=100):
        self.contexts = []
        self.max_contexts = max_contexts

    def add_context(self, domain, question, answer):
        context = {
            'domain': domain,
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
        self.contexts.append(context)
        if len(self.contexts) > self.max_contexts:
            self.contexts.pop(0)

    def get_relevant_context(self, domain, question):
        relevant = []
        for ctx in self.contexts:
            if ctx['domain'] == domain:
                if self._calculate_similarity(question, ctx['question']) > 0.3:
                    relevant.append(ctx)
        return relevant[-3:]

    def _calculate_similarity(self, txt1, txt2):
        set1 = set(txt1.lower().split())
        set2 = set(txt2.lower().split())
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union) if union else 0

memory = ContextualMemory()

# =========================
# Détection de domaine et outils
# =========================

class DomainSpecializer:
    def __init__(self):
        self.domains = {
            'mathematics': {
                'keywords': ['calcul', 'équation', 'mathématiques', 'géométrie', 'algèbre', 'statistiques'],
                'prompt_prefix': "En tant qu'expert en mathématiques, résolvez ce problème étape par étape:",
                'tools': ['calculator', 'equation_solver']
            },
            'science': {
                'keywords': ['physique', 'chimie', 'biologie', 'expérience', 'formule', 'loi'],
                'prompt_prefix': "En tant que scientifique expert, expliquez avec précision:",
                'tools': ['formula_calculator', 'unit_converter']
            },
            'programming': {
                'keywords': ['code', 'programmation', 'python', 'javascript', 'algorithme', 'debug'],
                'prompt_prefix': "En tant qu'expert en programmation, fournissez une solution complète:",
                'tools': ['code_analyzer', 'syntax_checker']
            },
            'medicine': {
                'keywords': ['médecine', 'santé', 'symptôme', 'diagnostic', 'traitement', 'maladie'],
                'prompt_prefix': "En tant que professionnel de santé (à des fins éducatives uniquement):",
                'tools': ['symptom_checker', 'drug_interaction']
            },
            'finance': {
                'keywords': ['finance', 'investissement', 'bourse', 'économie', 'budget', 'crédit'],
                'prompt_prefix': "En tant qu'expert financier, analysez:",
                'tools': ['financial_calculator', 'market_analyzer']
            },
            'history': {
                'keywords': ['histoire', 'historique', 'guerre', 'civilisation', 'empire', 'révolution'],
                'prompt_prefix': "En tant qu'historien expert, expliquez avec contexte:",
                'tools': ['timeline_generator', 'fact_checker']
            },
            'literature': {
                'keywords': ['littérature', 'poésie', 'roman', 'auteur', 'analyse', 'critique'],
                'prompt_prefix': "En tant qu'expert littéraire, analysez:",
                'tools': ['text_analyzer', 'style_detector']
            }
        }

    def detect_domain(self, text):
        text_lower = text.lower()
        scores = {}
        for domain, config in self.domains.items():
            score = sum(1 for kw in config['keywords'] if kw in text_lower)
            if score > 0:
                scores[domain] = score
        if scores:
            return max(scores, key=scores.get)
        return 'general'

    def get_specialized_prompt(self, domain, question):
        if domain in self.domains:
            prefix = self.domains[domain]['prompt_prefix']
            return f"{prefix}\n\nQuestion: {question}\n\nRéponse détaillée:"
        return f"Question: {question}\n\nRéponse:"

domain_specializer = DomainSpecializer()

# =========================
# Modèles et réseaux neuronaux
# =========================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class AttentionLayer(nn.Module):
    def __init__(self, in_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(in_dim, in_dim // heads)
        self.key = nn.Linear(in_dim, in_dim // heads)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
        attn_weights = self.softmax(scores)
        out = torch.matmul(attn_weights, V)
        return out

class DeepResNet(nn.Module):
    def __init__(self, input_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.res1 = ResidualBlock(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.res2 = ResidualBlock(256, 256)
        self.attention = AttentionLayer(256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.res1(x.unsqueeze(2)).squeeze(2)
        x = F.relu(self.fc2(x))
        x = self.res2(x.unsqueeze(2)).squeeze(2)
        x = self.attention(x.unsqueeze(1)).squeeze(1)
        x = F.relu(self.fc3(x))
        return self.out(x)

class ModelTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataloader, epochs=20):
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
        print("Entraînement terminé.")

    def evaluate(self, X_test, y_test):
        with torch.no_grad():
            outputs = self.model(X_test)
            preds = torch.argmax(outputs, dim=1)
            accuracy = (preds == y_test).float().mean().item()
            print(f"Précision sur test : {accuracy * 100:.2f}%")
            return accuracy

# =========================
# Sauvegarde et chargement
# =========================

class CheckpointManager:
    def __init__(self, model, dir='checkpoints'):
        self.model = model
        self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def save(self, filename='model.pth'):
        path = os.path.join(self.dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Modèle sauvegardé : {path}")

    def load(self, filename='model.pth'):
        path = os.path.join(self.dir, filename)
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print(f"Modèle chargé : {path}")
        else:
            print(f"Fichier non trouvé : {path}")

# =========================
# API Flask et interface web
# =========================

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link rel="shortcut icon" href="{{ url_for('static', filename='images/logo_darklab.jpg') }}" type="image/jpeg">
<title>DarkLab AI</title>
<style>
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  body {
    background: linear-gradient(135deg, #000, #2c003e);
    background-size: 400% 400%;
    animation: neonPulse 10s ease-in-out infinite;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #fff;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
  }

  @keyframes neonPulse {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
  }

  /* Section d'entrée utilisateur */
  #userInfo {
    background: rgba(0,0,0,0.7);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 0 20px #9400d3;
    width: 90%;
    max-width: 400px;
  }
  #userInfo h3 {
    margin-bottom: 10px;
    text-align: center;
    color: #e0e0e0;
  }
  #userInfo input {
    width: 100%;
    margin: 5px 0;
    padding: 10px;
    border-radius: 20px;
    border: none;
    outline: none;
    font-size: 1rem;
    background-color: #222;
    color: #fff;
    transition: box-shadow 0.3s;
  }
  #userInfo input:focus {
    box-shadow: 0 0 10px #9400d3;
  }
  #userInfo button {
    width: 100%;
    padding: 10px;
    margin-top: 10px;
    background: linear-gradient(135deg, #8a2be2, #9400d3);
    color: #fff;
    border: none;
    border-radius: 20px;
    cursor: pointer;
    font-size: 1rem;
    box-shadow: 0 0 10px #ff00ff, 0 0 20px #9400d3;
    transition: background 0.3s, box-shadow 0.3s, transform 0.2s;
  }
  #userInfo button:hover {
    background: linear-gradient(135deg, #9400d3, #8a2be2);
    box-shadow: 0 0 15px #ff00ff, 0 0 25px #9400d3;
    transform: scale(1.05);
  }

  /* Header */
  .header {
    text-align: center;
    margin-bottom: 20px;
  }
  .header h1 {
    font-size: 2rem;
    color: #e0e0e0;
    text-shadow: 0 0 10px #9400d3, 0 0 20px #9400d3;
  }

  /* Logo stylisé en cercle */
  .logo-circle {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    object-fit: cover;
    box-shadow: 0 0 15px rgba(148, 0, 211, 0.7);
    background-color: rgba(255,255,255,0.2);
    margin-bottom: 10px;
  }

  .domain-indicator {
    margin-top: 10px;
    font-size: 1rem;
    color: #d0d0d0;
    text-shadow: 0 0 5px #9400d3;
  }

  /* Chat container */
  .chat-container {
    width: 100%;
    max-width: 700px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .chat-history {
    background-color: rgba(0,0,0,0.6);
    padding: 15px;
    border-radius: 10px;
    width: 100%;
    min-height: 300px;
    overflow-y: auto;
    box-shadow: inset 0 0 20px #9400d3;
    margin-bottom: 10px;
  }
  /* scrollbar stylé */
  .chat-history::-webkit-scrollbar {
    width: 8px;
  }
  .chat-history::-webkit-scrollbar-thumb {
    background-color: #9400d3;
    border-radius: 4px;
  }

  /* Messages styles */
  .message {
    margin-bottom: 10px;
    padding: 10px;
    border-radius: 8px;
    max-width: 90%;
    line-height: 1.4;
  }
  .user-message {
    background-color: #222;
    align-self: flex-end;
    color: #fff;
    box-shadow: 0 0 10px #8a2be2;
  }
  .ai-message {
    background-color: #333;
    align-self: flex-start;
    color: #fff;
    box-shadow: 0 0 10px #9400d3;
  }
  /* Meta info in message */
  .message-meta {
    font-size: 0.75rem;
    margin-top: 5px;
    color: #ccc;
    font-style: italic;
  }

  /* Loader spinner */
  .loading {
    display: none;
    align-items: center;
    justify-content: center;
    margin: 10px 0;
  }
  .spinner {
    border: 4px solid rgba(255,255,255,0.2);
    border-top: 4px solid #9400d3;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin-right: 10px;
  }
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  /* Input area */
  .input-container {
    display: flex;
    width: 100%;
    max-width: 700px;
    margin-top: 10px;
  }
  #questionInput {
    flex: 1; /* occupe tout l'espace disponible */
    min-width: 0; /* évite que le contenu dépasse */
    padding: 10px 15px;
    border-radius: 20px;
    border: none;
    outline: none;
    font-size: 1rem;
    background-color: #222;
    color: #fff;
    box-shadow: inset 0 0 8px #9400d3;
    transition: all 0.2s ease; /* pour effet fluide si besoin */
  }
  #questionInput:focus {
    box-shadow: inset 0 0 12px #9400d3;
  }
  #sendBtn {
    margin-left: 10px;
    padding: 10px 20px;
    background: linear-gradient(135deg, #8a2be2, #9400d3);
    border: none;
    border-radius: 20px;
    cursor: pointer;
    font-size: 1rem;
    color: #fff;
    box-shadow: 0 0 10px #ff00ff, 0 0 20px #9400d3;
    transition: background 0.3s, box-shadow 0.3s, transform 0.2s;
  }
  #sendBtn:hover {
    background: linear-gradient(135deg, #9400d3, #8a2be2);
    box-shadow: 0 0 15px #ff00ff, 0 0 25px #9400d3;
    transform: scale(1.05);
  }

  /* Stats */
  .stats {
    display: flex;
    justify-content: space-around;
    width: 100%;
    max-width: 700px;
    margin-top: 20px;
  }
  .stat {
    background: rgba(0,0,0,0.7);
    padding: 10px 15px;
    border-radius: 10px;
    box-shadow: 0 0 15px #9400d3;
    text-align: center;
    flex: 1;
    margin: 0 5px;
  }
  .stat div {
    font-size: 1.5rem;
    font-weight: bold;
    color: #fff;
  }
  @media(max-width: 600px){
    .stats {
      flex-direction: column;
      align-items: center;
    }
    .stat {
      margin: 10px 0;
    }
  }
</style>
<script>
  let userName = '';
  let userEmail = '';

  function saveUserInfo() {
    userName = document.getElementById('userName').value.trim();
    userEmail = document.getElementById('userEmail').value.trim();
    if (userName && userEmail) {
      document.getElementById('userInfo').style.display='none';
      addMessage(`Bonjour ${userName} ! Comment puis-je vous aider aujourd'hui ?`, 'ai');
    } else {
      alert('Veuillez remplir votre nom et email.');
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('questionInput').addEventListener('keypress', (e) => {
      if (e.key === 'Enter') sendQuestion();
    });
    document.getElementById('sendBtn').addEventListener('click', sendQuestion);
  });

  async function sendQuestion() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    if (!question || !userName) return;

    addMessage(question, 'user');
    input.value = '';
    document.getElementById('loading').style.display = 'flex';

    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, userName, userEmail })
      });
      const data = await response.json();
      addMessage(data.response, 'ai', data);
      updateStats(data);
      updateDomainIndicator(data.domain);
    } catch (err) {
      addMessage('Désolé, une erreur est survenue.', 'ai');
    } finally {
      document.getElementById('loading').style.display = 'none';
    }
  }

  function addMessage(text, sender, data=null) {
    const chat = document.getElementById('chatHistory');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;
    let content = `<div>${text}</div>`;
    if (data && sender==='ai') {
      const conf = Math.round(data.confidence*100);
      const tools = data.tools_used.length ? data.tools_used.join(', ') : 'Aucun';
      content += `<div class="message-meta">Domaine: ${data.domain} | Confiance: ${conf}% | Outils: ${tools}</div>`;
    }
    msgDiv.innerHTML = content;
    chat.appendChild(msgDiv);
    chat.scrollTop = chat.scrollHeight;
  }

  let questionCount = 0;
  let domainSet = new Set();
  let totalConfidence = 0;
  function updateStats(data) {
    questionCount++;
    domainSet.add(data.domain);
    totalConfidence += data.confidence;
    document.getElementById('domainCount').textContent = domainSet.size;
    document.getElementById('questionCount').textContent = questionCount;
    document.getElementById('avgConfidence').textContent = Math.round((totalConfidence / questionCount)*100)+'%';
  }

  function updateDomainIndicator(domain) {
    const labels = {
      'mathematics': 'Mathématiques 🔢',
      'science': 'Sciences 🔬',
      'programming': 'Programmation 💻',
      'medicine': 'Médecine 🏥',
      'finance': 'Finance 💰',
      'history': 'Histoire 📚',
      'literature': 'Littérature 📖',
      'psychology': 'Psychologie 🧠',
      'technology': 'Technologie 🚀',
      'art': 'Art 🎨',
      'general': 'Général 🌐'
    };
    document.getElementById('currentDomain').textContent = `Domaine actuel: ${labels[domain] || domain}`;
  }
</script>
</head>
<body>
<!-- Section pour entrer nom et email -->
<div id="userInfo">
  <h3>Veuillez renseigner votre nom et email pour commencer</h3>
  <input type="text" id="userName" placeholder="Votre nom" />
  <input type="email" id="userEmail" placeholder="Votre email" />
  <button onclick="saveUserInfo()">Valider</button>
</div>

<!-- Interface principale -->
<div class="header">
  <!-- Logo stylisé en cercle avec transparence -->
 <img src="{{ url_for('static', filename='images/logo_darklab.jpg') }}" alt="Logo DarkLab"  class="logo-circle">
  <h1>🧠 DarkLab AI</h1>
  <div class="domain-indicator" id="currentDomain">Domaine actuel: Général</div>
</div>
<div class="chat-container">
  <div class="chat-history" id="chatHistory">
    <!-- Message initial -->
    <div class="ai-message message">
      <div>👋 Bonjour! Je suis DarkLab votre assistant IA</div>
      <div class="message-meta">Confiance: 100% | Outils disponibles</div>
    </div>
  </div>
  <div class="loading" id="loading" style="display:none;">
    <div class="spinner"></div>
    Analyse en cours...
  </div>
  <div class="input-container">
    <input type="text" id="questionInput" placeholder="Posez votre question dans n'importe quel domaine..." />
    <button id="sendBtn">Envoyer</button>
  </div>
  <div class="stats">
    <div class="stat">
      <div id="domainCount">0</div>
      <div>Domaines explorés</div>
    </div>
    <div class="stat">
      <div id="questionCount">0</div>
      <div>Questions traitées</div>
    </div>
    <div class="stat">
      <div id="avgConfidence">0%</div>
      <div>Confiance moyenne</div>
    </div>
  </div>
</div>
</body>
</html>
 """

# =========================
# Gestion calendrier
# =========================

calendar_events = []

def manage_calendar(question):
    question_lower = question.lower()
    if 'ajoute' in question_lower and 'événement' in question_lower:
        try:
            parts = question.split('événement')
            if len(parts) > 1:
                event_title = parts[1].strip()
                event_date = datetime.now().isoformat()
                calendar_events.append({'title': event_title, 'date': event_date})
                return f"Événement '{event_title}' ajouté à votre calendrier le {event_date}."
            else:
                return "Désolé, je n'ai pas pu extraire le titre de l'événement."
        except Exception:
            return "Erreur lors de l'ajout de l'événement."
    elif 'voir' in question_lower and 'événements' in question_lower:
        if calendar_events:
            events_str = '\n'.join([f"{ev['title']} le {ev['date']}" for ev in calendar_events])
            return f"Voici vos événements :\n{events_str}"
        else:
            return "Vous n'avez pas d'événements programmés."
    else:
        return "Je peux vous aider à gérer votre calendrier. Dites-moi ce que vous souhaitez faire."

# =========================
# API Groq
# =========================

def call_groq_api(question):
    url = 'https://api.groq.com/search'  # À adapter selon ton API
    headers = {
        'Authorization': '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',
        'Content-Type': 'application/json'
    }
    data = {
        'query': question
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result.get('answer', 'Aucune réponse trouvée.')
        else:
            return f"Erreur API Groq : {response.status_code} {response.text}"
    except Exception as e:
        return f"Erreur lors de l'appel API Groq : {str(e)}"

# =========================
# Route principale
# =========================

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

# =========================
# Route API
# =========================

@app.route('/api/ask', methods=['POST'])
def api():
    data = request.get_json()
    question = data.get('question', '')
    userName = data.get('userName', 'Utilisateur')
    userEmail = data.get('userEmail', '')

    # Détection si recherche Groq
    if 'recherche' in question.lower() or 'bibliothèque' in question.lower():
        response_text = call_groq_api(question)
    elif 'événement' in question.lower() or 'calendrier' in question.lower():
        response_text = manage_calendar(question)
    else:
        response_text = get_response_from_dan(question)

    # Détection du domaine
    domain = domain_specializer.detect_domain(question)
    confidence = 0.9  # Peut être ajusté en fonction de ton algorithme
    tools_used = []  # À remplir si tu utilises des outils

    # Ajout à la mémoire
    memory.add_context(domain, question, response_text)

    return jsonify({
        'response': response_text,
        'domain': domain,
        'confidence': confidence,
        'tools_used': tools_used
    })

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# =========================
# Fonction réponse GPT
# =========================

def get_response_from_dan(question):
    question_lower = question.lower()

    creation_keywords = [
        'qui t\'a créé', 'who created you',  # français, anglais
        '谁创造了你',  # chinois
        '誰があなたを作った',  # japonais
        'quien te creó',  # espagnol
        '谁开发了你',  # chinois simplifié
        'qui vous a créé',  # français
        'wer hat dich erstellt',  # allemand
        '谁开发了你',  # chinois
        'qui t\'a construit',  # français
        '谁建造了你'  # chinois
        'who made you' #anglais
    ]

    # Vérifie si l'un des mots-clés est présent dans la question
    if any(keyword in question_lower for keyword in creation_keywords):
        return "Je suis créé par Minkande Eba'a Efandene Daniel Darnel."

    # Sinon, utilise l'API GPT
    prompt = (
        "Tu es un expert en synthèse d’informations. "
        "Lorsque je te donne des données, présente-les de manière claire et lisible, "
        "en évitant l’utilisation d’étoiles, traits d’union ou autres caractères spéciaux. "
        "Formate la réponse en paragraphes ou listes simples, en te concentrant sur la lisibilité.\n\n"
        f"Question : {question}"
    )
    try:
        if not OPENAI_API_KEY:
            return "Clé API non configurée."
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Erreur API : {str(e)}")
        return f"Erreur lors de la requête : {str(e)}"
def train_model():
    print("Lancement de l'entraînement du réseau neuronale...")

    # Création de données factices (à remplacer par ton vrai dataset)
    input_dim = 128  # correspond à ton input_dim dans DeepResNet
    output_dim = 10  # à ajuster selon ton problème

    # Génère des données aléatoires pour l'exemple
    X_train = torch.randn(1000, input_dim)
    y_train = torch.randint(0, output_dim, (1000,))

    # Création du DataLoader
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Instanciation du modèle
    model = DeepResNet(input_dim=input_dim, output_dim=output_dim)

    # Instanciation du trainer
    trainer = ModelTrainer(model, lr=config.learning_rate)

    # Entraînement
    trainer.train(dataloader, epochs=config.epochs)

    # Sauvegarde du modèle
    checkpoint = CheckpointManager(model)
    checkpoint.save('model_final.pth')

    print("Entraînement terminé et modèle sauvegardé.")
# =========================
# Démarrage
# =========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script pour entraîner ou lancer le serveur.')
    parser.add_argument('--train', action='store_true', help='Lancer l’entraînement du réseau de neurones.')
    args = parser.parse_args()

    if args.train:
        train_model()
    else:
        app.run(host='0.0.0.0', port=8000)
