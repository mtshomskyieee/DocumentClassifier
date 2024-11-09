import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.util import ngrams
import string
import re

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK resources
        nltk_resources = ['punkt', 'stopwords', 'wordnet']
        for resource in nltk_resources:
            try:
                print(f"Checking NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Error downloading {resource}: {str(e)}")
                raise
        
        # Initialize stemmers and lemmatizer
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.porter_stemmer = PorterStemmer()
            self.snowball_stemmer = SnowballStemmer('english')
            self.tweet_tokenizer = TweetTokenizer()
        except Exception as e:
            print(f"Error initializing text processing components: {str(e)}")
            raise
        
        # Default preprocessing options
        self.config = {
            'lowercase': True,
            'remove_numbers': True,
            'remove_punctuation': True,
            'remove_whitespace': True,
            'remove_urls': True,
            'remove_emails': True,
            'tokenizer': 'word',  # Options: 'word', 'sentence', 'tweet'
            'stemmer': 'lemmatizer',  # Options: 'lemmatizer', 'porter', 'snowball', None
            'remove_stopwords': True,
            'custom_stopwords': set(),
            'min_word_length': 2,
            'max_word_length': 100,
            'ngram_range': (1, 1)  # (min_n, max_n)
        }
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Error loading stopwords: {str(e)}")
            self.stop_words = set()
            raise
        
    def set_config(self, **kwargs):
        """Update preprocessing configuration"""
        self.config.update(kwargs)
        if 'custom_stopwords' in kwargs:
            try:
                self.stop_words = set(stopwords.words('english')).union(kwargs['custom_stopwords'])
            except Exception as e:
                print(f"Error updating stopwords: {str(e)}")
                raise
    
    def clean_text(self, text):
        """Basic text cleaning"""
        try:
            if self.config['lowercase']:
                text = text.lower()
            
            if self.config['remove_urls']:
                text = re.sub(r'http\S+|www\S+|https\S+', '', text)
                
            if self.config['remove_emails']:
                text = re.sub(r'\S+@\S+', '', text)
                
            if self.config['remove_numbers']:
                text = re.sub(r'\d+', '', text)
                
            if self.config['remove_punctuation']:
                text = text.translate(str.maketrans('', '', string.punctuation))
                
            if self.config['remove_whitespace']:
                text = ' '.join(text.split())
                
            return text
        except Exception as e:
            print(f"Error in text cleaning: {str(e)}")
            raise
    
    def tokenize(self, text):
        """Tokenize text based on selected tokenizer"""
        try:
            if self.config['tokenizer'] == 'sentence':
                return sent_tokenize(text)
            elif self.config['tokenizer'] == 'tweet':
                return self.tweet_tokenizer.tokenize(text)
            else:  # word tokenizer
                return word_tokenize(text)
        except Exception as e:
            print(f"Error in tokenization: {str(e)}")
            raise
    
    def apply_stemming(self, token):
        """Apply selected stemming/lemmatization method"""
        try:
            if self.config['stemmer'] == 'porter':
                return self.porter_stemmer.stem(token)
            elif self.config['stemmer'] == 'snowball':
                return self.snowball_stemmer.stem(token)
            elif self.config['stemmer'] == 'lemmatizer':
                return self.lemmatizer.lemmatize(token)
            return token
        except Exception as e:
            print(f"Error in stemming/lemmatization: {str(e)}")
            raise
    
    def generate_ngrams(self, tokens):
        """Generate n-grams from tokens"""
        try:
            min_n, max_n = self.config['ngram_range']
            all_ngrams = []
            for n in range(min_n, max_n + 1):
                all_ngrams.extend([' '.join(gram) for gram in ngrams(tokens, n)])
            return all_ngrams
        except Exception as e:
            print(f"Error generating n-grams: {str(e)}")
            raise
    
    def preprocess(self, text):
        """Main preprocessing pipeline"""
        try:
            # Clean text
            text = self.clean_text(text)
            
            # Tokenize
            tokens = self.tokenize(text)
            
            # Filter tokens
            tokens = [token for token in tokens 
                     if self.config['min_word_length'] <= len(token) <= self.config['max_word_length']]
            
            # Remove stopwords
            if self.config['remove_stopwords']:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            # Apply stemming/lemmatization
            if self.config['stemmer']:
                tokens = [self.apply_stemming(token) for token in tokens]
            
            # Generate n-grams if required
            if self.config['ngram_range'] != (1, 1):
                tokens = self.generate_ngrams(tokens)
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in preprocessing pipeline: {str(e)}")
            raise
