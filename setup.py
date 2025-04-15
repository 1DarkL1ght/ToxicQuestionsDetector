import nltk

def download():
    nltk.download('stopwords')
    nltk.download('popular')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')

if __name__ == "__main__":
    print("Downloading nltk data...")
    download()
    print("Download complete.")
    print("You can now run the main.py file to test the model.")
    print("If you encounter any issues, please check the nltk documentation for troubleshooting.")
