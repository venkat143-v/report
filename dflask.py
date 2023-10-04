from flask import *
import os
import base64
import secrets
import pdftextextract as pte
import imagetextextract as ite

secret_key = secrets.token_hex(16)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
@app.route("/bulk", methods=['GET', 'POST'])


def bulk():
    uploaded_filename = None  # Initialize the variable to None

    if request.method == "POST":
        file = request.files['data']
        if file:
            na = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], na))
            uploaded_filename=na
            sfile=None
            def extract_text_from_file(file_path):
                file_extension = os.path.splitext(file_path)[-1].lower()
                if file_extension == '.pdf':
                    sfile=pte.ext_from_pdf(file_path)
                elif file_extension  in ('.jpg', '.jpeg', '.png'):
                    sfile=ite.extract_text_from_image(file_path)
                else:
                    flash("Unsupported file format. Please upload a PDF, JPG, or PNG file.", "error")
                    return redirect(request.url)
                return sfile
            file_path = r"static/data/"+na
            uploaded_filename=extract_text_from_file(file_path)
            os.remove(file_path)
    return render_template('index.html', uploaded_filename=uploaded_filename)


@app.route('/view_pdf/<pdf_filename>')
def view_pdf(pdf_filename):
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)

    try:
        # Open the PDF file and read its content
        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
        
        response = Response(pdf_data, content_type='application/pdf')
        os.remove(pdf_path)
        return response
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")
        return "Failed to load PDF document"

if __name__ == '__main__':
    app.secret_key = secret_key
    app.run(debug=True)



