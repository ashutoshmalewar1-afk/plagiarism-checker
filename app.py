import os
import itertools
import nltk
from flask import Flask, request, render_template, send_file
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt
from fpdf import FPDF

nltk.download('punkt')

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')


def read_files(files):

    texts = []
    filenames = []

    for file in files:

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())

        filenames.append(file.filename)

    return texts, filenames


def semantic_similarity(texts):

    embeddings = model.encode(texts)

    sim_matrix = cosine_similarity(embeddings)

    return sim_matrix


def tfidf_similarity(texts):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(texts)

    sim_matrix = cosine_similarity(vectors)

    return sim_matrix


def calculate_plagiarism(texts, filenames):

    semantic_sim = semantic_similarity(texts)
    tfidf_sim = tfidf_similarity(texts)

    results = []

    for i, j in itertools.combinations(range(len(texts)), 2):

        score = (semantic_sim[i][j] * 0.7) + (tfidf_sim[i][j] * 0.3)

        percent = round(score * 100, 2)

        if percent > 60:
            risk = "High"
        elif percent > 30:
            risk = "Medium"
        else:
            risk = "Low"

        results.append({
            "file1": filenames[i],
            "file2": filenames[j],
            "score": percent,
            "risk": risk
        })

    return results


def generate_graph(results):

    G = nx.Graph()

    for r in results:

        if r["score"] > 20:
            G.add_edge(r["file1"], r["file2"], weight=r["score"])

    plt.figure(figsize=(8,6))

    pos = nx.spring_layout(G)

    nx.draw(G, pos, with_labels=True, node_size=3000,
            node_color="skyblue")

    labels = nx.get_edge_attributes(G,'weight')

    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    os.makedirs("static", exist_ok=True)

    path = "static/plagiarism_graph.png"

    plt.savefig(path)

    plt.close()

    return path


def generate_pdf(results):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=14)

    pdf.cell(200,10,"Plagiarism Detection Report", ln=True, align="C")

    pdf.ln(10)

    pdf.set_font("Arial", size=12)

    for r in results:

        line = f"{r['file1']} vs {r['file2']} -> {r['score']}% ({r['risk']} Risk)"

        pdf.cell(200,10,line, ln=True)

    pdf_path = "plagiarism_report.pdf"

    pdf.output(pdf_path)

    return pdf_path


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/check", methods=["POST"])
def check():

    files = request.files.getlist("files")

    texts, filenames = read_files(files)

    results = calculate_plagiarism(texts, filenames)

    graph = generate_graph(results)

    pdf = generate_pdf(results)

    return render_template("result.html",
                           results=results,
                           graph=graph,
                           pdf=pdf)


@app.route("/download/<path:filename>")
def download(filename):

    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)