import io
import os.path

import flask
from flask import Flask, jsonify, Response
from flask import make_response
from werkzeug.wsgi import FileWrapper

from pdf_to_docx import convert_pdf_to_docx

app = Flask(__name__)


@app.route("/convert-pdf", methods=["GET"])
def handle_pdf_file():
    raw_data = flask.request.get_data()
    convert_pdf_to_docx(raw_data)
    if os.path.exists("PDF_TO_DOCX.docx"):
        with open("PDF_TO_DOCX.docx", "rb") as f:
            w = FileWrapper(io.BytesIO(f.read()))
            return Response(w, mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            direct_passthrough=True)
    else:
        return None


@app.errorhandler(404)
def not_found():
    return make_response(jsonify({"error": "Not found"}), 404)


@app.errorhandler(500)
def server_error():
    return make_response(jsonify({"server error": "Something went wrong..."}), 500)


if __name__ == '__main__':
    app.run()
