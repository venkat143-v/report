from flask import*
import os
import pdfplumber
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pytesseract
from PIL import Image
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
import cv2
import fitz 
import docx  
import numpy as np
import tensorflow as tf
from tensorflow import keras
import spacy

from flask import Flask, render_template, Response
import os
import fitz  # PyMuPDF
import base64


app = Flask(__name__)
UPLOAD_FOLDER = 'static/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

    
# Function to extract text from PDF files
def extract_text_from_pdf(pdf_path):
    text = []
    pdf_document = fitz.open(pdf_path)
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        page_text = page.get_text()
        text.append(page_text)
    return text


# Function to preprocess text using spaCy
def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.text for token in doc]
    processed_text = " ".join(tokens)
    return processed_text

def disease_predict(file_path):
    def extract_text(file_path):
        file_extension = os.path.splitext(file_path)[-1].lower()
        if file_extension=='.pdf':
            text = []
            pdf_document = fitz.open(file_path)
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text()
                text.append(page_text)
            return text
        else:
            image = cv2.imread(file_path)
            text_from_image = pytesseract.image_to_string(image)
            return [text_from_image]

    file_extension = os.path.splitext(file_path)[-1].lower()

    combined_text = extract_text(file_path)
    max_sequence_length = 100
    vocabulary_size = 10000
    embedding_dim = 32
    
    # Move the tokenizer creation outside of the loop
    tokenizer = tf.keras.layers.TextVectorization(max_tokens=vocabulary_size, output_mode="int")
    tokenizer.adapt(combined_text)  # Adapt to the combined text
    ans=[]
    for page_num, processed_text in enumerate(combined_text):
        sequences = tokenizer(processed_text)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences([sequences], maxlen=max_sequence_length)

        model = keras.Sequential([
            keras.layers.Input(shape=(max_sequence_length,)),
            keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dim),
            keras.layers.LSTM(128),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        predicted_disease_prob = model(padded_sequences)  # Call the model directly on tensors
        predicted_disease_label = 1 if predicted_disease_prob >= 0.5 else 0
        ans.append(predicted_disease_label)
    return ans


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
                isdisease=disease_predict(file_path)
            ##    print(isdisease)
                if file_extension == '.pdf':
                    with pdfplumber.open(file_path) as pdf:
                        pdf_pages = pdf.pages
                        pdf_pages_count = len(pdf_pages)
                        pdf_filename = "generated_report.pdf"  # Name of the output PDF file
                        sfile = pdf_filename
                        oppath=r"static/data/"+pdf_filename
                        # Create a PdfPages object to save the generated images
                        pdf_pages_to_save = PdfPages(oppath)
                        di=0
                        for page_number, page in enumerate(pdf_pages):
                            page_text = page.extract_text()
                            lines = page_text.split('\n')
                            l = lines
                            n = len(lines)
                            i = 0
                            dis = []
                            res = []
                            uni = []
                            uniflag=False
                            while i < (n - 1):
                                b = lines[i].lower().split()
                                if "test" in b[:4]:
                                    if("units" in b):
                                        uniflag=True
                                    break
                                i += 1
                            i += 1
                            while i < n:
                                b = lines[i].split()
            ##                    print(b)
                                if(b[0].lower() == "male" or b[0].lower() == "female"):
                                    i += 1
                                    continue
                                j = 0
                                s = ""
                                flag = False
                                while j < len(b) - 1:
                                    f = [(True) for i in "0123456789" if i in b[j + 1]]
                                    if True in f:
                                        s += b[j]
                                        flag = True
                                        break
                                    else:
                                        s += b[j] + " "
                                    j += 1
                                if (s.lower() == 'page' or s == '|' or 'note' in s.lower()or ("patientreportcategory" in s.lower())
                                                    or("repeats are accepted" in s.lower()) or ("request within" in s.lower())
                                    or ("customer care tel no" in s.lower()) or ("patientreportscsuper" in s.lower())):
                                    i += 1
                                    continue
                                alr=[]
                                if flag == False:
                                    s += b[j]
                                else:
                                    s1 = ""
                                    cnt=0
                                    for al in b[j + 1]:
                                        if al=='.':
                                            cnt+=1
                                        if cnt==2:
                                            break
                                        if al.isdigit() or al == '.':
                                            s1 += al
                                    if s not in dis:
                                        dis.append(s)
                                        res.append(s1)
                                        if(uniflag==True):
                                            uni.append(b[j+2])
                                        else:
                                            uni.append(b[-1])
                                    else:
                                        i+=1
                                        continue
                                i += 1
                            comlis = [[float(res[i]), dis[i][:20]] for i in range(len(dis))]
            ##                comlis.sort()
                            if(len(comlis)==0):
                                di+=1
                                continue
            ##                print(uni)
                            newdis = [comlis[i][1] for i in range(len(comlis))]
                            newres = [comlis[i][0] for i in range(len(comlis))]
            ##                print(len(newdis),len(uni))
                            plt.rcParams["figure.figsize"] = [16, 9]
                            fig, ax = plt.subplots()
                            ax.barh(newdis, newres)
                            for s in ['top', 'bottom', 'left', 'right']:
                                ax.spines[s].set_visible(False)
                            ax.xaxis.set_ticks_position('none')
                            ax.yaxis.set_ticks_position('none')
                            ax.xaxis.set_tick_params(pad=5)
                            ax.yaxis.set_tick_params(pad=10)
                            ax.grid(True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

                            ax.invert_yaxis()

                            un=0
                            for i in ax.patches:
                                plt.text(i.get_width() + 0.2, i.get_y() + 0.5, str(round(i.get_width(), 2))+" "+uni[un],
                                         fontsize=15, fontweight='bold', color='pink')
                                un+=1

                            ax.set_ylabel("Test",fontsize=15,fontweight='bold',color='darkblue')
                            ax.set_xlabel("Result",fontsize=15,fontweight='bold',color='darkblue')
                            ax.set_title("Test Report",loc='center',fontsize=15,fontweight='bold')
                            if(isdisease[di]==1):
                                ax.set_title("This person has disease",loc='right',fontsize=15,fontweight='bold',color='red')
                            else:
                                ax.set_title("This person is healthy",loc='right',fontsize=15,fontweight='bold',color='green')
                            di+=1
                            pdf_pages_to_save.savefig(fig)

                else:
                    pdf_filename = "generated_report.pdf"  # Name of the output PDF file
                    # Create a PdfPages object to save the generated images
                    sfile = pdf_filename
                    oppath=r"static/data/"+pdf_filename
                    # Create a PdfPages object to save the generated images
                    pdf_pages_to_save = PdfPages(oppath)
                    img = Image.open(file_path)
                    text = pytesseract.image_to_string(img)
                    lines = text.split('\n')
                    l = lines
                    n = len(lines)
                    i = 0
                    dis = []
                    res = []
                    
                    while i < (n - 1):
                        b = lines[i].lower().split()
                        if(len(b)>0):
                            if "test" in b[:4]:
                                if("units" in b):
                                    uniflag=True
                                break
                        i += 1
                    i += 1
                    while i < n:
                        b = lines[i].split()
            ##            print(b)
                        if(len(b)>0):
                            if(b[0].lower() == "male" or b[0].lower() == "female"):
                                i += 1
                                continue
                            j = 0
                            s = ""
                            flag = False
                            if(b[0].lower()=="blood" or b[0].lower()=="serum"):
                                    j+=1
                            while j < len(b) - 1:
                                f = [(True) for i in "0123456789" if i in b[j + 1]]
                                if True in f:
                                    s += b[j]
                                    flag = True
                                    break
                                else:
                                    s += b[j] + " "
                                j += 1
                            if (s.lower() == 'page' or s == '|' or 'note' in s.lower()or ("patientreportcategory" in s.lower())
                                                or("repeats are accepted" in s.lower()) or ("request within" in s.lower())
                                or ("customer care tel no" in s.lower()) or ("patientreportscsuper" in s.lower())):
                                i += 1
                                continue
                            if flag == False:
                                s += b[j]
                                dis.append(s)
                                res.append(0)
                            else:
                                s1 = ""
                                cnt=0
                                for al in b[j + 1]:
                                    if al=='.':
                                        cnt+=1
                                    if cnt==2:
                                            break
                                    if al.isdigit() or al == '.':
                                        s1 += al
                                if s not in dis:
                                    dis.append(s)
                                    res.append(s1)
                                else:
                                    i+=1
                                    continue
                        i += 1
                    comlis = [[float(res[i]), dis[i][:20]] for i in range(len(dis))]
                    comlis.sort()
                    newdis = [comlis[i][1] for i in range(len(comlis))]
                    newres = [comlis[i][0] for i in range(len(comlis))]
                    plt.rcParams["figure.figsize"] = [16, 9]
                    fig, ax = plt.subplots()
                    ax.barh(newdis, newres)
                    for s in ['top', 'bottom', 'left', 'right']:
                        ax.spines[s].set_visible(False)
                    ax.xaxis.set_ticks_position('none')
                    ax.yaxis.set_ticks_position('none')
                    ax.xaxis.set_tick_params(pad=5)
                    ax.yaxis.set_tick_params(pad=10)
                    ax.grid(True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

                    ax.invert_yaxis()


                    for i in ax.patches:
                        plt.text(i.get_width() + 0.2, i.get_y() + 0.5, str(round(i.get_width(), 2)),
                                    fontsize=15, fontweight='bold', color='pink')

                    ax.set_ylabel("Test",fontsize=15,fontweight='bold',color='darkblue')
                    ax.set_xlabel("Result",fontsize=15,fontweight='bold',color='darkblue')
                    if(isdisease[0]==1):
                        ax.set_title("This person has disease",loc='right',fontsize=15,fontweight='bold',color='red')
                    else:
                        ax.set_title("This person is healthy",loc='right',fontsize=15,fontweight='bold',color='green')

                    pdf_pages_to_save.savefig(fig)
                # Close the PDF file
                pdf_pages_to_save.close()
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
    app.run(debug=True)



