import os
from flask import Flask, request, Response, jsonify


from detect_objects import analyze_jpeg_bytes, format_text_output

DEFAULT_PORT = 8080

# Use a relative path or an absolute path based on the file's location
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pages'))
app = Flask(__name__, static_folder=template_dir, static_url_path="")


from flask import send_from_directory


@app.get("/health")
def health():
    return "ok", 200


@app.get("/")
def index():
    return send_from_directory("pages", "index.html")


@app.post("/upload")
def upload_and_analyze():
    if "file" not in request.files:
        return jsonify(error="Missing form field 'file'"), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify(error="No file selected"), 400

    if (f.mimetype or "").lower() != "image/jpeg":
        return jsonify(error="Only image/jpeg is accepted", got=f.mimetype), 415

    jpeg_bytes = f.read()
    if not jpeg_bytes:
        return jsonify(error="Empty upload"), 400

    try:
        payload = analyze_jpeg_bytes(jpeg_bytes)
        text = format_text_output(payload)
        return Response(text, mimetype="text/plain")
    except Exception as e:
        return Response(
            f"ERROR analyzing image:\n{e}",
            status=500,
            mimetype="text/plain",
        )





if __name__ == "__main__":
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    app.run(host="0.0.0.0", port=port, debug=False)

