import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import PhotoImage, Scrollbar
from PIL import Image, ImageTk
import threading

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    stop_word = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_word]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    preprocessed_text = ' '.join(words)
    return preprocessed_text


new_ds = pickle.load(open('preprocess_data.pkl', 'rb'))

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def suggest_sections(complaint, dataset, min_suggestions=5):
    preprocessed_complaint = preprocess_text(complaint)
    complaint_embedding = model.encode(preprocessed_complaint)
    section_embedding = model.encode(dataset['Combo'].tolist())
    similarities = util.pytorch_cos_sim(complaint_embedding, section_embedding)[0]
    similarity_threshold = 0.2
    relevant_indices = []
    while len(relevant_indices) < min_suggestions and similarity_threshold > 0:
        relevant_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        similarity_threshold -= 0.05
    sorted_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)
    suggestions = dataset.iloc[sorted_indices][['Description', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court', 'Combo']].to_dict(orient='records')
    return suggestions

def get_suggestion():
    complaint = complaint_entry.get()
    suggestions = suggest_sections(complaint, new_ds)
    output_text.delete(1.0, END)
    if suggestions:
        output_text.insert(END, "Suggested IPS Sections are:\n", 'header')
        output_text.tag_config('header', font=('Helvetica', 12, 'bold'))
        for suggestion in suggestions:
            output_text.insert(END, f"Description: {suggestion['Description']}\n", 'bold')
            output_text.insert(END, f"Offense: {suggestion['Offense']}\n", 'bold')
            output_text.insert(END, f"Punishment: {suggestion['Punishment']}\n", 'bold')
            output_text.insert(END, "----------------------------------------------------------\n")


def get_suggestion_with_loading():
    loading_label.pack(pady=10)
    output_text.delete(1.0, END)
    thread = threading.Thread(target=lambda: [get_suggestion(), loading_label.pack_forget()])
    thread.start()

root = tb.Window(themename="litera")
root.title("IPC SECTION SUGGESTER - By Sakshi Puria")
root.geometry("1200x900")  


background_image = Image.open("FIRprojpic.png")
background_image = background_image.resize((800, 900), Image.LANCZOS)
background_photo = ImageTk.PhotoImage(background_image)


left_frame = tb.Frame(root, width=500, height=700)
left_frame.pack(side=LEFT, fill=Y)
right_frame = tb.Frame(root, padding=20, width=500, height=700)
right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

background_label = tb.Label(left_frame, image=background_photo) # left frame
background_label.place(x=0, y=0, relwidth=1, relheight=1)

welcome_label = tb.Label(right_frame, text="Welcome Back!", font=('Helvetica', 16, 'bold')) # Right frame 
welcome_label.pack(pady=10)

instruction_label = tb.Label(right_frame, text="Enter Crime Description", font=('Helvetica', 12))
instruction_label.pack(pady=5)

complaint_entry = tb.Entry(right_frame, width=50, bootstyle=PRIMARY)
complaint_entry.pack(pady=5)

suggest_button = tb.Button(right_frame, text="Get Suggestion", command=get_suggestion_with_loading, bootstyle=(PRIMARY, OUTLINE))
suggest_button.pack(pady=10)

loading_label = tb.Label(right_frame, text="Loading...", font=('Helvetica', 12, 'italic')) # loading label

output_frame = tb.Frame(right_frame)
output_frame.pack(fill=BOTH, expand=True)

output_scrollbar = Scrollbar(output_frame)
output_scrollbar.pack(side=RIGHT, fill=Y)

output_text = tb.Text(output_frame, width=50, height=20, yscrollcommand=output_scrollbar.set)
output_text.pack(side=LEFT, fill=BOTH, expand=True)
output_scrollbar.config(command=output_text.yview)

root.mainloop()
