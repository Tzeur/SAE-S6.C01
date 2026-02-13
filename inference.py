"""
Script d'Inférence pour Évaluation
===================================
Ce script permet de prédire la polarité d'avis Yelp avec:
1. Le meilleur modèle ML (SVM + TF-IDF)
2. Le meilleur modèle Deep Learning (MLP + TF-IDF)

Usage:
    python inference.py "Chemin/vers/fichier_test.csv"
    python inference.py --text "The food was amazing!"
"""

import argparse
import json
import pickle
import pandas as pd
import torch
import torch.nn as nn
import os
import sys

# Colors for terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# Import MLP from src package
from src.models import MLP


def load_models():
    """Charge les modèles et vectorizers sauvegardés."""
    print(f"{Colors.BLUE}Chargement des modèles...{Colors.RESET}")
    
    # Load config
    with open(os.path.join(MODELS_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Load TF-IDF vectorizer
    with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
    
    # Load ML model
    with open(os.path.join(MODELS_DIR, 'best_ml_model.pkl'), 'rb') as f:
        ml_model = pickle.load(f)
    
    # Load DL model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl_model = MLP(input_dim=config['input_dim'])
    dl_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_dl_model.pth'), map_location=device))
    dl_model.to(device)
    dl_model.eval()
    
    print(f"{Colors.GREEN}Modèles chargés avec succès!{Colors.RESET}")
    return tfidf, ml_model, dl_model, config, device


def predict_single(text, tfidf, ml_model, dl_model, config, device):
    """Prédit la polarité d'un seul texte."""
    inverse_map = {int(k): v for k, v in config['inverse_polarity_map'].items()}
    
    # Transform text
    X_tfidf = tfidf.transform([text])
    
    # ML prediction
    ml_pred = ml_model.predict(X_tfidf)[0]
    ml_label = inverse_map[ml_pred]
    
    # DL prediction
    X_tensor = torch.tensor(X_tfidf.toarray(), dtype=torch.float32).to(device)
    with torch.no_grad():
        dl_output = dl_model(X_tensor)
        dl_pred = torch.argmax(dl_output, dim=1).cpu().numpy()[0]
    dl_label = inverse_map[dl_pred]
    
    return ml_label, dl_label


def predict_file(filepath, tfidf, ml_model, dl_model, config, device):
    """Prédit la polarité pour un fichier CSV/JSON."""
    print(f"\n{Colors.BLUE}Lecture du fichier: {filepath}{Colors.RESET}")
    
    # Read file
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.json') or filepath.endswith('.jsonl'):
        df = pd.read_json(filepath, lines=filepath.endswith('.jsonl'))
    else:
        raise ValueError("Format non supporté. Utilisez .csv, .json ou .jsonl")
    
    # Find text column
    text_col = None
    for col in ['text', 'review', 'content', 'comment', 'Text', 'Review']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print(f"{Colors.YELLOW}Colonnes disponibles: {list(df.columns)}{Colors.RESET}")
        text_col = df.columns[0]
        print(f"{Colors.YELLOW}  → Utilisation de la première colonne: '{text_col}'{Colors.RESET}")
    
    print(f"{Colors.BLUE}{len(df)} lignes à traiter...{Colors.RESET}")
    
    inverse_map = {int(k): v for k, v in config['inverse_polarity_map'].items()}
    
    # Transform all texts
    texts = df[text_col].fillna('').astype(str).tolist()
    X_tfidf = tfidf.transform(texts)
    
    # ML predictions
    ml_preds = ml_model.predict(X_tfidf)
    df['ml_prediction'] = [inverse_map[p] for p in ml_preds]
    
    # DL predictions
    X_tensor = torch.tensor(X_tfidf.toarray(), dtype=torch.float32).to(device)
    with torch.no_grad():
        dl_outputs = dl_model(X_tensor)
        dl_preds = torch.argmax(dl_outputs, dim=1).cpu().numpy()
    df['dl_prediction'] = [inverse_map[p] for p in dl_preds]
    
    # Save results
    output_path = filepath.replace('.csv', '_predictions.csv').replace('.json', '_predictions.csv')
    df.to_csv(output_path, index=False)
    print(f"{Colors.GREEN}Résultats sauvegardés: {output_path}{Colors.RESET}")
    
    # Display summary
    print(f"\n{Colors.BOLD}📈 Résumé des prédictions:{Colors.RESET}")
    print(f"  ML (SVM + TF-IDF):")
    for label in ['positive', 'neutral', 'negative']:
        count = (df['ml_prediction'] == label).sum()
        pct = count / len(df) * 100
        print(f"    - {label}: {count} ({pct:.1f}%)")
    
    print(f"\n  DL (MLP + TF-IDF):")
    for label in ['positive', 'neutral', 'negative']:
        count = (df['dl_prediction'] == label).sum()
        pct = count / len(df) * 100
        print(f"    - {label}: {pct:.1f}%)")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Inférence de polarité pour avis Yelp")
    parser.add_argument('input', nargs='?', help="Chemin vers fichier CSV/JSON ou texte en mode --text")
    parser.add_argument('--text', '-t', action='store_true', help="Mode texte unique")
    parser.add_argument('--interactive', '-i', action='store_true', help="Mode interactif")
    
    args = parser.parse_args()
    
    # Load models
    try:
        tfidf, ml_model, dl_model, config, device = load_models()
    except FileNotFoundError as e:
        print(f"{Colors.RED}Erreur: Modèles non trouvés. Exécutez d'abord le notebook 2_prediction_models.ipynb{Colors.RESET}")
        print(f"   Détail: {e}")
        sys.exit(1)
    
    print(f"\n{Colors.BOLD}Modèle ML: {config['ml_model']} + {config['ml_representation']}{Colors.RESET}")
    print(f"{Colors.BOLD}Modèle DL: {config['dl_model']} + {config['dl_representation']}{Colors.RESET}")
    
    # Interactive mode
    if args.interactive or (args.input is None and not args.text):
        print(f"\n{Colors.BOLD}🔮 Mode Interactif - Tapez 'quit' pour quitter{Colors.RESET}")
        while True:
            try:
                text = input(f"\n{Colors.BLUE}Entrez un avis:{Colors.RESET} ")
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                ml_label, dl_label = predict_single(text, tfidf, ml_model, dl_model, config, device)
                
                # Color based on prediction
                colors = {'positive': Colors.GREEN, 'neutral': Colors.YELLOW, 'negative': Colors.RED}
                print(f"\n  {Colors.BOLD}ML:{Colors.RESET} {colors[ml_label]}{ml_label.upper()}{Colors.RESET}")
                print(f"  {Colors.BOLD}DL:{Colors.RESET} {colors[dl_label]}{dl_label.upper()}{Colors.RESET}")
            except KeyboardInterrupt:
                break
        print(f"\n{Colors.BLUE}Au revoir!{Colors.RESET}")
    
    # Single text mode
    elif args.text:
        ml_label, dl_label = predict_single(args.input, tfidf, ml_model, dl_model, config, device)
        print(f"\n{Colors.BOLD}Texte:{Colors.RESET} {args.input[:100]}...")
        print(f"\n{Colors.BOLD}Prédictions:{Colors.RESET}")
        print(f"  ML (SVM + TF-IDF): {ml_label.upper()}")
        print(f"  DL (MLP + TF-IDF): {dl_label.upper()}")
    
    # File mode
    else:
        if not os.path.exists(args.input):
            print(f"{Colors.RED}Fichier non trouvé: {args.input}{Colors.RESET}")
            sys.exit(1)
        predict_file(args.input, tfidf, ml_model, dl_model, config, device)


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}   SYSTÈME D'INFÉRENCE - ANALYSE YELP{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    main()
