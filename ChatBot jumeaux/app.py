from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import PyPDF2
import re
import logging
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialiser l'application Flask
app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="")

# Récupérer les variables d'environnement
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("VERTEX_LOCATION", "europe-west9")
pdf_paths = os.getenv("PDF_PATHS", "./documents/livre-blanc.pdf, ./documents/BTCPres.pdf, ./documents/GTH_slide.pdf, \
                      ./documents/projet_promethee.pdf, ./documents/cahier_acteur_SNBC-PPE.pdf").split(",")

# Vérifier si les informations essentielles sont présentes
if not project_id:
    print("ATTENTION: GOOGLE_CLOUD_PROJECT n'est pas défini. Définissez cette variable d'environnement.")

# Initialiser Vertex AI seulement si le projet est défini
model = None
if project_id:
    try:
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel('gemini-1.5-pro')
        print(f"Vertex AI initialisé avec succès pour le projet {project_id}")
    except Exception as e:
        print(f"Erreur lors de l'initialisation de Vertex AI: {e}")


# Variable globale pour stocker les chunks de tous les PDFs
all_pdf_chunks = []

def chunk_pdf(pdf_file, chunk_size=4000, overlap=1000):
    """
    Divise un fichier PDF en chunks de texte avec chevauchement. Amelioration sur plusieurs pdf
    """
    try:
        # Lire le PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        # Extraire le texte de chaque page
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Vérifier que la page contient du texte
                text += page_text + " "
        
        # Nettoyer le texte
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Créer les chunks avec chevauchement
        chunks = []
        if len(text) <= chunk_size:
            chunks.append(text)
        else:
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                
                # Si nous ne sommes pas au début du texte et que nous ne sommes pas
                # à la fin, essayons de couper à un espace pour ne pas couper un mot
                if start > 0 and end < len(text):
                    # Chercher le dernier espace dans le chunk
                    last_space = text[start:end].rfind(' ')
                    if last_space != -1:
                        end = start + last_space
                
                chunks.append(text[start:end])
                
                # Déplacer le point de départ pour le prochain chunk, en tenant compte du chevauchement
                start = end - overlap if end < len(text) else end
        
        print(f"Chunking terminé pour {pdf_file}: {len(chunks)} chunks créés")
        return chunks
    except Exception as e:
        print(f"Erreur lors du chunking du PDF {pdf_file}: {e}")
        return []
    


def select_chunks_embeddings(question, chunks, top_n=8):
    """
    Sélectionne les chunks les plus pertinents en utilisant les embeddings ET la recherche de mots-clés.
    """
    # Extraction des mots-clés de la question
    keywords = re.findall(r'\b\w+\b', question.lower())
    exact_matches = []
    
    # Recherche de correspondances exactes
    for i, chunk in enumerate(chunks):
        # Calculer un score pour chaque chunk basé sur les mots-clés présents
        keyword_score = sum(1 for keyword in keywords if keyword.lower() in chunk.lower())
        if keyword_score > 0:
            exact_matches.append((i, keyword_score))
    
    # Charger le modèle
    model_path = "./paraphrase-multilingual-mpnet-base-v2/"
    model = SentenceTransformer(model_path)
    # model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # Modèle multilingue plus performant
    # model = SentenceTransformer('paraphrase-distilroberta-base-v1')  # Modèle plus léger
    # model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    # model =  SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') # environ 90 Mo
    # Calculer les embeddings
    question_embedding = model.encode(question)
    chunk_embeddings = model.encode(chunks)
    
    # Calculer les similarités cosinus
    similarities = np.dot(chunk_embeddings, question_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    
    # Combiner les scores d'embedding et de mots-clés
    combined_scores = similarities.copy()
    for idx, keyword_score in exact_matches:
        combined_scores[idx] += keyword_score * 0.1  # Pondération des scores de mots-clés
    
    # Sélectionner les chunks les plus similaires
    top_indices = np.argsort(combined_scores)[-top_n:][::-1]
    selected_chunks = [chunks[i] for i in top_indices]
    
    return selected_chunks, combined_scores[top_indices]

def check_for_btc_keywords(question):
    """
    Vérifie si la question contient des mots-clés liés à BTC.
    """
    keywords = ["btc", "biotechnologies consultants"]
    return any(keyword.lower() in question.lower() for keyword in keywords)


def generate_btc_specific_response(question, context):
    """
    Génère une réponse spécifique pour les questions concernant BTC
    en y incorporant les informations clés requises.
    """
    # Information spécifique à inclure systématiquement pour BTC
    btc_key_info = """
BTC se spécialise dans la production de ressources stratégiques et d'énergie renouvelable (CH4, H2, CO2, électricité, engrais, eau azotée, métaux).
BTC fait partie des 24 projets ayant répondu à l'AMI orienté autour de la gazéification hydrothermale et constitue l'une des entreprises pionnières en la matière.
"""
    
    # Créer un prompt qui force l'inclusion de ces informations
    enhanced_prompt = f"""
Vous êtes un assistant travaillant chez BTC (Biotechnologies Consultants).

Question posée: {question}

Contexte extrait des documents:
{context}

Informations clés à inclure OBLIGATOIREMENT dans votre réponse:
{btc_key_info}
"""
    
    return enhanced_prompt


# Route pour servir les fichiers statiques
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('.', path)

# Ajoutez cette route spécifique pour l'image
@app.route('/Rubie_portrait.png')
def serve_image():
    return send_from_directory('.', 'Rubie_portrait.png')

@app.route('/blue-concentric-circles.jpg')
def serve_background():
    return send_from_directory('.', 'ciel.jpg')

# Charger les chunks au démarrage
@app.before_first_request
def load_pdf_chunks():
    global all_pdf_chunks
    all_pdf_chunks = []
    
    for pdf_path in pdf_paths:
        pdf_path = pdf_path.strip()
        if os.path.exists(pdf_path):
            print(f"Chargement du PDF: {pdf_path}")
            chunks = chunk_pdf(pdf_path)
            all_pdf_chunks.extend(chunks)
            print(f"PDF chargé: {pdf_path} - {len(chunks)} chunks ajoutés")
        else:
            print(f"ATTENTION: Le fichier PDF {pdf_path} n'existe pas")
    
    print(f"Total des chunks chargés: {len(all_pdf_chunks)}")

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/generate', methods=['POST'])
def generate():
    if not model:
        return jsonify({'error': 'Vertex AI n\'est pas configuré correctement. Vérifiez les variables d\'environnement.'}), 500
    
    global all_pdf_chunks
    
    # Paramètres pour une génération plus déterministe
    generation_config = {
        "temperature": 0.1,  # Très bas pour être plus déterministe
        "top_p": 0.9,
        "max_output_tokens": 1024,  # Limiter la longueur de la réponse
    }
    
    data = request.json
    prompt = data.get('prompt', '')
    language = data.get('language', 'FR')  # Paramètre de langue avec FR par défaut
    
    # Afficher la question pour débogage
    print(f"\033[31m{str(prompt)} (Langue: {language})\033[0m")

    if not prompt:
        return jsonify({'error': 'Le prompt est requis'}), 400
    
    try:
        # Sélectionner les chunks pertinents
        relevant_chunks, relevance_scores = select_chunks_embeddings(prompt, all_pdf_chunks, top_n=5)
        selected_text = "\n".join(relevant_chunks)
        
        # Extraire le chunk le plus pertinent (celui avec le score de pertinence le plus élevé)
        most_relevant_chunk = relevant_chunks[0]  # Le premier chunk est le plus pertinent

        # Dictionnaire des instructions de langue
        language_instructions = {
            'FR': "Répondez en français.",
            'ES': "Responda en español.",
            'IT': "Risponda in italiano.",
            'DE': "Antworten Sie auf Deutsch.",
            'GB': "Answer in English.",
            'JP': "日本語で答えてください。",
            'Suiss': "Répondez en suisse wallon."
        }
        
        # Obtenir l'instruction de langue ou utiliser le français par défaut
        lang_instruction = language_instructions.get(language, language_instructions['FR'])

        # Vérifier si la question concerne BTC
        if check_for_btc_keywords(prompt):
            # Utiliser le prompt spécifique à BTC avec l'instruction de langue
            btc_prompt = generate_btc_specific_response(prompt, selected_text)
            final_prompt = f"{btc_prompt}\n\n{lang_instruction}"
            
            final_response = model.generate_content(
                final_prompt,
                generation_config=generation_config
            )
        else:
            # Utiliser le prompt standard pour les autres questions avec l'instruction de langue
            final_prompt = f"""Voici des extraits pertinents d'un document:
{selected_text}
Extrait le plus pertinent:
{most_relevant_chunk}
Question: {prompt}

Répondez à la question en vous basant uniquement sur les extraits fournis. Si l'information est insuffisante, précisez-le clairement.
Citez clairement l'extrait le plus pertinent qui répond à la question dans votre réponse, entre des quotes "" et dans le format markdown italique.

{lang_instruction}"""
            
            final_response = model.generate_content(
                final_prompt,
                generation_config=generation_config
            )
            
        print(f"\033[32m{str(final_response.text)}\033[0m")
        
        # Créer une réponse structurée avec l'extrait le plus pertinent
        response_data = {
            'response': final_response.text,
            'most_relevant_chunk': most_relevant_chunk,
            'relevance_score': float(relevance_scores[0]),  # Convertir en float pour la sérialisation JSON
            'language': language  # Inclure la langue utilisée dans la réponse
        }
        
        return jsonify(response_data)  # Retourner la réponse

    except Exception as e:
        print(f"Erreur lors de la génération de la réponse: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'pdf_count': len(pdf_paths), 'chunks_count': len(all_pdf_chunks)})

if __name__ == '__main__':
    print("utilisation de la version 1.2 de l'application")
    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)