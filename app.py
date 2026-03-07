import os
from flask import Flask, render_template, request, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.pdfgen import canvas

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors)[0][1]
    return round(similarity * 100, 2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/check", methods=["POST"])
def check():
    file1 = request.files.get("file1")
    file2 = request.files.get("file2")

    if not file1 or not file2:
        return "Please upload both files."

    path1 = os.path.join(UPLOAD_FOLDER, file1.filename)
    path2 = os.path.join(UPLOAD_FOLDER, file2.filename)

    file1.save(path1)
    file2.save(path2)

    with open(path1, "r", encoding="utf-8", errors="ignore") as f:
        text1 = f.read()

    with open(path2, "r", encoding="utf-8", errors="ignore") as f:
        text2 = f.read()

    similarity = calculate_similarity(text1, text2)

    pdf_path = "plagiarism_report.pdf"
    c = canvas.Canvas(pdf_path)
    c.drawString(100, 750, "Plagiarism Report")
    c.drawString(100, 720, f"Similarity Score: {similarity}%")
    c.save()

    return render_template("result.html", similarity=similarity)


@app.route("/download")
def download():
    return send_file("plagiarism_report.pdf", as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)