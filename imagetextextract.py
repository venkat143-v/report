import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pytesseract
from PIL import Image
import cv2
import dispredict as dip
s1file=None
def extract_text_from_image(file_path):
    isdisease=dip.disease_predict(file_path)
    pdf_filename = "generated_report.pdf"  # Name of the output PDF file
    # Create a PdfPages object to save the generated images
    s1file = pdf_filename
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
    pdf_pages_to_save.close()
    return s1file
    

