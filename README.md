# Guide d'Utilisation - Projet SAE S6.C01

## 🚀 Démarrage Rapide

### 1. Prérequis
```bash
pip install pandas numpy scikit-learn torch transformers sentence-transformers matplotlib seaborn tqdm
```

### 2. Exécuter l'Analyse Complète
Les notebooks sont déjà exécutés. Pour les relancer:
```bash
cd SAE
jupyter notebook notebooks/
```

### 3. Faire de l'Inférence (JOUR DE L'ÉVALUATION)

```bash
# Option 1: Texte unique
python inference.py --text "Your review text here"

# Option 2: Fichier de test
python inference.py path/to/test_file.csv

# Option 3: Mode interactif
python inference.py -i
```

---

## 📊 Résultats des Modèles

### Meilleur Modèle ML
- **SVM + TF-IDF**: ~90% accuracy sur polarité

### Meilleur Modèle DL
- **MLP + TF-IDF**: ~88% accuracy sur polarité

---

## 📁 Fichiers Importants

| Fichier | Description |
|---------|-------------|
| `inference.py` | Script d'inférence pour évaluation |
| `models/best_ml_model.pkl` | Modèle SVM sauvegardé |
| `models/best_dl_model.pth` | Modèle MLP sauvegardé |
| `models/tfidf_vectorizer.pkl` | Vectorizer TF-IDF |
| `notebooks/2_prediction_models_executed.ipynb` | Notebook Phase B complet |

---

## 🎯 Points du Barème Couverts

✅ N-grammes  
✅ TF-IDF  
✅ LLM Embeddings  
✅ 4 Modèles ML (LogReg, SVM, NaiveBayes, RandomForest)  
✅ 3 Modèles DL (MLP, CNN, BiLSTM)  
✅ Inférence optimale fonctionnelle  
✅ Inférence sur données de test  
