def clean_text(text):
    """
    Nettoie le texte d'entrée pour s'assurer qu'il est au bon format (str).
    Gère les valeurs nulles (NaN/None).
    """
    if text is None:
        return ""
    return str(text)

def preprocess_dataframe(df, text_col):
    """
    Prépare un DataFrame pour l'inférence en s'assurant que la colonne texte est propre.
    """
    return df[text_col].fillna('').astype(str).tolist()
