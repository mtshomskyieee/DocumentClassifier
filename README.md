# Document Similarity Analyzer

Welcome to the Document Similarity Analyzer, a streamlit application exercising document comparison techniques

## Screenshot
![img.png](img.png)
## Features

- **Preprocessing Options**: Clean and prepare your text with options like lowercase conversion, punctuation removal, and more. 
- **Tokenization and N-grams**: Choose how to split your text and decide on n-gram sizes. 
- **Clustering and Visualization**: Group similar documents and visualize them with style. See your documents form clusters .
- **Similarity Metrics**: Dive deep into the numbers with cosine, Euclidean, Jaccard, and centroid similarities. 
- **Export Results**: Take your findings with you in JSON or CSV format.

## Installation

1. Clone the repo: `git clone https://github.com/yourusername/DocumentClassifier.git`
2. Navigate to the project directory: `cd DocumentClassifier`
3. **Optional**: setup virtual environment `python3 -m venv venv` or `virtualenv venv` then `source venv/bin/activate`
4. Install the requirements: `pip install -r requirements.txt`
5. **REQUIRED** Download nltk data (once per install) `python download_nltk.py`

## Usage

1. Fire up the app: `streamlit run main.py`
2. Upload your documents or use the sample data.
3. Configure your preprocessing settings in the sidebar.
4. Analyze the clusters and similarity metrics.
5. Export the results and bask in the glory of your findings.

## Contributing

Feel like adding a new feature or fixing a bug? Fork the repo, make your changes, and submit a pull request. 

## License

This project is licensed under the MIT License.
Copyright (c) 2024 Michael Shomsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

