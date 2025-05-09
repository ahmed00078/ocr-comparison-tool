import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import tempfile
import time
import base64
from PIL import Image
import cv2
from ocr_comparator import OCRModelComparator

# Configuration de la page
st.set_page_config(
    page_title="OCR Comparateur - Données Environnementales",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fonctions utilitaires
def load_css():
    """Charge le CSS personnalisé."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #34495e;
            margin-top: 2rem;
        }
        .card {
            border-radius: 5px;
            background-color: #f8f9fa;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-card {
            background-color: #e8f4f8;
            border-left: 5px solid #3498db;
        }
        .success-msg {
            color: #2ecc71;
            font-weight: bold;
        }
        .stProgress .st-ey {
            background-color: #3498db;
        }
        .file-list {
            margin-top: 10px;
            padding-left: 20px;
        }
        .model-info {
            font-size: 0.9rem;
            color: #7f8c8d;
            margin-top: 5px;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #7f8c8d;
            font-size: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Affiche l'en-tête de l'application."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">Comparateur OCR pour Données Environnementales</h1>', unsafe_allow_html=True)
        st.markdown("""
        <p style="text-align: center;">
            Extrayez et analysez des données environnementales à partir de documents techniques
            en comparant différents moteurs OCR
        </p>
        """, unsafe_allow_html=True)

def create_temp_directory():
    """Crée un répertoire temporaire pour les fichiers de travail."""
    temp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "ground_truth"), exist_ok=True)
    return temp_dir

def save_uploaded_files(uploaded_files, temp_dir, file_type="images"):
    """Sauvegarde les fichiers téléchargés dans le répertoire temporaire."""
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, file_type, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return file_paths

def preprocess_image(image_path, options):
    """Prétraite une image en fonction des options sélectionnées."""
    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Impossible de lire l'image: {image_path}")
        return image_path
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    processed = gray.copy()
    
    # Appliquer les options de prétraitement sélectionnées
    if options.get("enhance_contrast", False):
        # Amélioration du contraste adaptatif (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
    
    if options.get("remove_noise", False):
        # Réduction du bruit
        processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)
    
    if options.get("binarize", False):
        # Binarisation adaptative
        processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    
    if options.get("deskew", False):
        # Détection et correction de l'inclinaison
        coords = np.column_stack(np.where(processed > 0))
        if len(coords) > 0:  # S'assurer qu'il y a des pixels à traiter
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Appliquer la rotation seulement si l'angle est significatif
            if abs(angle) > 0.5:
                (h, w) = processed.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                processed = cv2.warpAffine(processed, M, (w, h), 
                                      flags=cv2.INTER_CUBIC, 
                                      borderMode=cv2.BORDER_REPLICATE)
    
    # Sauvegarde de l'image prétraitée
    output_path = image_path.replace(".", "_preprocessed.")
    cv2.imwrite(output_path, processed)
    
    return output_path

def run_ocr_comparison(models_to_test, temp_dir, preprocessing_options, progress_bar):
    """Exécute la comparaison OCR avec les modèles sélectionnés."""
    
    # Prétraitement des images si activé
    if any(preprocessing_options.values()):
        progress_bar.progress(0.1)
        st.info("Prétraitement des images en cours...")
        
        image_files = os.listdir(os.path.join(temp_dir, "images"))
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(temp_dir, "images", img_file)
            processed_path = preprocess_image(img_path, preprocessing_options)
            # Remplacer l'image originale par l'image prétraitée
            if os.path.exists(processed_path) and processed_path != img_path:
                os.replace(processed_path, img_path)
            
            # Mettre à jour la barre de progression
            progress_percentage = 0.1 + (0.2 * (i + 1) / len(image_files))
            progress_bar.progress(progress_percentage)
    else:
        progress_bar.progress(0.3)
    
    # Initialiser et exécuter le comparateur OCR
    st.info("Initialisation des modèles OCR...")
    progress_bar.progress(0.4)
    
    comparator = OCRModelComparator(
        models_to_test=models_to_test,
        ground_truth_dir=os.path.join(temp_dir, "ground_truth"),
        test_images_dir=os.path.join(temp_dir, "images")
    )
    
    progress_bar.progress(0.5)
    st.info("Exécution des tests OCR en cours...")
    
    results = comparator.run_tests()
    progress_bar.progress(0.8)
    
    # Sauvegarder les résultats
    output_dir = os.path.join(temp_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    comparator.save_results(output_dir)
    comparator.generate_visualizations(output_dir)
    
    progress_bar.progress(1.0)
    st.success("Analyse OCR terminée!")
    
    return results, output_dir

def extract_environmental_metrics(text):
    """
    Extrait les métriques environnementales du texte OCR.
    Recherche les patterns comme "X kg CO2e", "X kWh", etc.
    """
    import re
    
    metrics = {
        "carbon_footprint": [],
        "energy_consumption": [],
        "recycled_content": []
    }
    
    # Recherche d'empreinte carbone (ex: "12.5 kg CO2e", "5 kg de CO2")
    carbon_pattern = r'(\d+(?:[,.]\d+)?)\s*(?:kg)?\s*(?:CO2e?|carbone)'
    carbon_matches = re.findall(carbon_pattern, text, re.IGNORECASE)
    metrics["carbon_footprint"] = [float(m.replace(',', '.')) for m in carbon_matches if m]
    
    # Recherche de consommation d'énergie (ex: "5 kWh", "100 Wh")
    energy_pattern = r'(\d+(?:[,.]\d+)?)\s*(?:k)?(?:W|w)(?:h|H)'
    energy_matches = re.findall(energy_pattern, text, re.IGNORECASE)
    metrics["energy_consumption"] = [float(m.replace(',', '.')) for m in energy_matches if m]
    
    # Recherche de contenu recyclé (ex: "30% recyclé", "contenu recyclé: 45%")
    recycled_pattern = r'(\d+(?:[,.]\d+)?)\s*%\s*(?:recyclé|recycled)'
    recycled_matches = re.findall(recycled_pattern, text, re.IGNORECASE)
    metrics["recycled_content"] = [float(m.replace(',', '.')) for m in recycled_matches if m]
    
    return metrics

def display_results(results, output_dir):
    """Affiche les résultats de la comparaison OCR."""
    
    st.markdown('<h2 class="sub-header">Résultats de la comparaison OCR</h2>', unsafe_allow_html=True)
    
    # 1. Afficher les visualisations générées
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card"><h3>Comparaison des performances</h3>', unsafe_allow_html=True)
        comparison_path = os.path.join(output_dir, "ocr_comparison.png")
        if os.path.exists(comparison_path):
            st.image(comparison_path, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card"><h3>Distribution des métriques</h3>', unsafe_allow_html=True)
        distributions_path = os.path.join(output_dir, "ocr_distributions.png")
        if os.path.exists(distributions_path):
            st.image(distributions_path, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 2. Afficher le tableau des métriques détaillées
    st.markdown('<div class="card"><h3>Métriques détaillées par modèle</h3>', unsafe_allow_html=True)
    metrics_path = os.path.join(output_dir, "ocr_metrics.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        
        # Ajouter une mise en forme conditionnelle pour faciliter la lecture
        def highlight_best(s, metric, is_lower_better=True):
            if is_lower_better:
                is_best = s == s.min()
            else:
                is_best = s == s.max()
            return ['background-color: rgba(46, 204, 113, 0.3)' if v else '' for v in is_best]
        
        # Appliquer le formatage
        styled_df = metrics_df.style.apply(highlight_best, metric='cer', is_lower_better=True, subset=['cer'])
        styled_df = styled_df.apply(highlight_best, metric='wer', is_lower_better=True, subset=['wer'])
        styled_df = styled_df.apply(highlight_best, metric='similarity', is_lower_better=False, subset=['similarity'])
        styled_df = styled_df.apply(highlight_best, metric='processing_time', is_lower_better=True, subset=['processing_time'])
        
        # Renommer les colonnes pour une meilleure présentation
        styled_df = styled_df.format({
            'cer': '{:.3f}',
            'wer': '{:.3f}',
            'similarity': '{:.3f}',
            'processing_time': '{:.2f} s'
        }).set_properties(**{
            'text-align': 'center'
        })
        
        st.dataframe(styled_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3. Afficher les textes extraits et les métriques environnementales
    st.markdown('<h2 class="sub-header">Texte extrait et métriques environnementales</h2>', unsafe_allow_html=True)
    
    # Créer des onglets pour chaque modèle
    if results:
        tabs = st.tabs([f"Modèle: {model}" for model in results.keys()])
        
        for i, model_name in enumerate(results.keys()):
            with tabs[i]:
                for image_file, text in results[model_name]['texts'].items():
                    with st.expander(f"Image: {image_file}", expanded=True):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("#### Texte extrait:")
                            st.text_area("", text, height=300, key=f"{model_name}_{image_file}_text")
                        
                        with col2:
                            st.markdown("#### Métriques environnementales extraites:")
                            
                            # Extraire les métriques environnementales
                            env_metrics = extract_environmental_metrics(text)
                            
                            # Afficher les métriques dans des cartes distinctes
                            for metric_name, values in env_metrics.items():
                                if values:
                                    st.markdown(f"""
                                    <div class="card metric-card">
                                        <h4>{metric_name.replace('_', ' ').title()}</h4>
                                        <p>Valeurs détectées: {', '.join([str(v) for v in values])}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            if not any(values for values in env_metrics.values()):
                                st.info("Aucune métrique environnementale détectée dans ce texte.")
                        
                        # Afficher les métriques OCR pour cette image
                        if 'metrics' in results[model_name] and image_file in results[model_name]['metrics']:
                            metrics = results[model_name]['metrics'][image_file]
                            st.markdown("#### Métriques OCR:")
                            metrics_cols = st.columns(3)
                            with metrics_cols[0]:
                                st.metric("Taux d'erreur caractères (CER)", f"{metrics['character_error_rate']:.3f}")
                            with metrics_cols[1]:
                                st.metric("Taux d'erreur mots (WER)", f"{metrics['word_error_rate']:.3f}")
                            with metrics_cols[2]:
                                st.metric("Similarité", f"{metrics['similarity_ratio']:.3f}")

def create_download_button(output_dir):
    """Crée un bouton pour télécharger les résultats."""
    # Créer un ZIP contenant les résultats
    import zipfile
    
    # Chemin du fichier ZIP temporaire
    zip_path = os.path.join(output_dir, "ocr_results.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Ajouter les fichiers de résultats
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file != "ocr_results.zip":  # Éviter de zipper le ZIP lui-même
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
    
    # Lire le fichier ZIP et créer un bouton de téléchargement
    with open(zip_path, "rb") as f:
        bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="ocr_results.zip" class="download-button">Télécharger tous les résultats</a>'
        st.markdown(f"""
        <div style="text-align: center; margin-top: 20px;">
            {href}
        </div>
        <style>
            .download-button {{
                background-color: #3498db;
                color: white;
                padding: 12px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            .download-button:hover {{
                background-color: #2980b9;
            }}
        </style>
        """, unsafe_allow_html=True)

def sidebar_config():
    """Configuration de la barre latérale avec les options."""
    st.sidebar.title("Configuration")
    
    # 1. Sélection des modèles OCR
    st.sidebar.subheader("Modèles OCR")
    tesseract = st.sidebar.checkbox("Tesseract OCR", value=True, help="Le moteur OCR open-source le plus connu")
    easyocr = st.sidebar.checkbox("EasyOCR", value=True, help="Basé sur les réseaux de neurones, simple à utiliser")
    trocr = st.sidebar.checkbox("TrOCR (Transformers)", value=False, help="Modèle OCR basé sur les transformers")
    paddleocr = st.sidebar.checkbox("PaddleOCR", value=False, help="Développé par Baidu, haut de gamme")
    
    models_to_test = []
    if tesseract: models_to_test.append('tesseract')
    if easyocr: models_to_test.append('easyocr')
    if trocr: models_to_test.append('trocr')
    if paddleocr: models_to_test.append('paddleocr')
    
    if not models_to_test:
        st.sidebar.warning("Veuillez sélectionner au moins un modèle OCR")
    
    # 2. Options de prétraitement
    st.sidebar.subheader("Prétraitement d'image")
    preprocessing = {}
    preprocessing["enhance_contrast"] = st.sidebar.checkbox("Amélioration du contraste", value=False, 
                                                           help="Améliore le contraste pour les petits caractères")
    preprocessing["remove_noise"] = st.sidebar.checkbox("Réduction du bruit", value=False,
                                                      help="Réduit le bruit d'image pour une meilleure précision")
    preprocessing["binarize"] = st.sidebar.checkbox("Binarisation adaptative", value=False,
                                                  help="Convertit l'image en noir et blanc de manière adaptative")
    preprocessing["deskew"] = st.sidebar.checkbox("Redressement automatique", value=False,
                                               help="Corrige l'inclinaison de document")
    
    # 3. Informations sur l'outil
    st.sidebar.markdown("---")
    with st.sidebar.expander("À propos de cet outil"):
        st.markdown("""
        **Comparateur OCR pour données environnementales**
        
        Cet outil fait partie du projet de stage:
        *"Création d'une base de données carbone pour une électronique plus écologique"* à l'INSA Rennes.
        
        Il compare différents moteurs OCR pour:
        - Extraire des données environnementales
        - Évaluer la précision de reconnaissance
        - Identifier le meilleur modèle pour votre cas d'usage
        """)
    
    # 4. Bouton d'aide rapide
    st.sidebar.markdown("---")
    with st.sidebar.expander("Aide rapide"):
        st.markdown("""
        **Guide d'utilisation:**
        
        1. Sélectionnez les modèles OCR à comparer
        2. Téléchargez vos images de documents
        3. (Optionnel) Téléchargez les fichiers de vérité terrain
        4. Configurez les options de prétraitement
        5. Lancez la comparaison
        6. Analysez les résultats et téléchargez-les
        
        **Formats supportés:**
        - Images: PNG, JPG, JPEG, TIFF
        - Vérité terrain: fichiers texte (TXT)
        """)
    
    return models_to_test, preprocessing

def main():
    """Fonction principale de l'application Streamlit."""
    # Charger le CSS et afficher l'en-tête
    load_css()
    display_header()
    
    # Configuration de la barre latérale
    models_to_test, preprocessing_options = sidebar_config()
    
    # Création d'un répertoire temporaire pour les fichiers
    temp_dir = create_temp_directory()
    
    # Section 1: Téléchargement d'images
    st.markdown('<h2 class="sub-header">1. Téléchargement d\'images</h2>', unsafe_allow_html=True)
    uploaded_images = st.file_uploader(
        "Téléchargez des images de documents environnementaux",
        type=["png", "jpg", "jpeg", "tiff"],
        accept_multiple_files=True,
        help="Téléchargez des images de documents contenant des données environnementales"
    )
    
    if uploaded_images:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.success(f"{len(uploaded_images)} image(s) téléchargée(s)")
        
        # Afficher des miniatures des images téléchargées
        image_cols = st.columns(min(4, len(uploaded_images)))
        for i, uploaded_file in enumerate(uploaded_images[:4]):  # Limiter à 4 miniatures
            with image_cols[i % 4]:
                st.image(uploaded_file, caption=uploaded_file.name, width=150)
        
        if len(uploaded_images) > 4:
            st.info(f"+ {len(uploaded_images) - 4} autres images")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sauvegarder les images
        save_uploaded_files(uploaded_images, temp_dir, "images")
    
    # Section 2: Téléchargement de vérités terrain (facultatif)
    st.markdown('<h2 class="sub-header">2. Vérités terrain (facultatif)</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        Les fichiers de vérité terrain contiennent le texte exact attendu pour chaque image.
        Ils permettent de calculer des métriques de précision comme le taux d'erreur de caractères (CER).
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_gt = st.file_uploader(
        "Téléchargez les fichiers texte de vérité terrain",
        type=["txt"],
        accept_multiple_files=True,
        help="Les noms des fichiers doivent correspondre aux images (sans l'extension)"
    )
    
    if uploaded_gt:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.success(f"{len(uploaded_gt)} fichier(s) de vérité terrain téléchargé(s)")
        
        # Afficher les noms des fichiers
        gt_names = [f.name for f in uploaded_gt]
        st.markdown('<div class="file-list">', unsafe_allow_html=True)
        for name in gt_names:
            st.markdown(f"- {name}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sauvegarder les fichiers de vérité terrain
        save_uploaded_files(uploaded_gt, temp_dir, "ground_truth")
    
    # Section 3: Lancement de la comparaison OCR
    st.markdown('<h2 class="sub-header">3. Lancement de la comparaison OCR</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_button = st.button(
            "Lancer la comparaison des modèles OCR",
            disabled=not uploaded_images or not models_to_test,
            help="Lance l'analyse OCR avec les modèles sélectionnés"
        )
    
    # Exécuter la comparaison OCR si le bouton est cliqué
    if start_button:
        # Afficher une barre de progression
        progress_text = "Analyse en cours. Veuillez patienter..."
        my_bar = st.progress(0, text=progress_text)
        
        # Lancer la comparaison OCR
        results, output_dir = run_ocr_comparison(
            models_to_test=models_to_test,
            temp_dir=temp_dir,
            preprocessing_options=preprocessing_options,
            progress_bar=my_bar
        )
        
        # Afficher les résultats
        display_results(results, output_dir)
        
        # Créer un bouton de téléchargement des résultats
        create_download_button(output_dir)
    
    # Pied de page avec le nom de l'auteur et le projet
    st.markdown("""
    <div class="footer">
        <p>Outil OCR pour l'extraction de données environnementales | Ahmed Sidi Mohamed | INSA Rennes 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Nettoyage du répertoire temporaire à la fermeture de l'application
    # Note: Cela ne fonctionne pas toujours comme prévu dans Streamlit
    # car le script peut continuer à s'exécuter.
    # Nous gardons cette fonction pour le principe.
    def cleanup():
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    import atexit
    atexit.register(cleanup)

if __name__ == "__main__":
    main()