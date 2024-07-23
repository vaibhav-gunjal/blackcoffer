import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string



def extract_text(url):
    try:
        # send a get request 
        response = requests.get(url)

        if response.status_code == 200:
            # parsing object
            soup = BeautifulSoup(response.content,'html.parser')

            # title extract
            title_tag = soup.find('h1', class_='entry-title')
            if title_tag:
                title = title_tag.get_text(strip=True)
            else:
                title = 'No title found'
            
            # Find the main content container
            main_content = soup.find('div', class_='td-post-content') # article in this class       
            text_list=[]
            if main_content:
                # get all paragraphs within the main content 
                for element in main_content.find_all(['p', 'ol']):
                    if element.name == 'p':
                        text_list.append(element.get_text(strip=True))
                        
                    elif element.name == 'ol':
                        # Print list item text
                        for li in element.find_all('li'):
                            text_list.append(li.get_text(strip=True))
            
            # Combine all parts into a single text
                content = '\n'.join(text_list)
                return title, content
            
            else:
                return "content not found", ""
            
        elif response.status_code == 404:
            return f"Error 404: Page not found", ""
                            
        else:
            return "Main content not found.",""
        
    except Exception as e:
        return f"Error: {str(e)}", ""

# provide url from excel file
excel_file='Input.xlsx'
df=pd.read_excel(excel_file,sheet_name='Sheet1')

# Create a directory to store text files
output_dir = "extracted_texts"
os.makedirs(output_dir, exist_ok=True)

# itrate url through for loop
for index ,row in df.iterrows():
    url=row['URL']
    text_file=row['URL_ID']
    
    title, content = extract_text(url)

    if title != "Failed to extract webpage":
        # Generate filename on url_id 
        filename = os.path.join(output_dir, f"{text_file}.txt")
        
        # Add text in .txt file
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"{title}\n")
            file.write(content)
            print(f"Data saved to {filename}")
    else:
        print(f"Failed to retrieve data for {url}")
    
    print('*'*40)



# Load positive and negative word lists
with open('MasterDictionary/positive-words.txt', 'r') as f:
    positive_words = set(f.read().splitlines())
with open('MasterDictionary/negative-words.txt', 'r') as f:
    negative_words = set(f.read().splitlines())

# Load stop words
stop_words = set()
for filename in os.listdir('StopWords'):
    with open(os.path.join('StopWords', filename), 'r') as f:
        stop_words.update(f.read().splitlines())



def sentimental_analysis(text):
    # Tokenization
    text = re.sub(r'[^\w\s.]', '', text)
    tokens = word_tokenize(text.lower()) 
    # cleaning of text using stopwords
    cleaned_tokens = [word for word in tokens if word not in stop_words]

    # calculating positive and negative score as per statemnet
    positive_score = 0   
    for word in cleaned_tokens:
        if word in positive_words:
            positive_score += 1

    negative_score = 0
    for word in cleaned_tokens:
        if word in negative_words:
            negative_score += 1

    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / ((len(cleaned_tokens)) + 0.000001)

    return {
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score
           }

def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("es"):
        count -= 1
    
    if word.endswith("ed"):
        count -= 1

    if count == 0:
        count += 1

    return count



# Analysis of Readability
def readability_analysis(text):
    # Tokenization
    sentences = sent_tokenize(text)
    # removing punctuvation
    text = re.sub(r'[^\w\s.]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)

    complex_word_count = 0
    for word in words:
        if syllable_count(word)>2:
            complex_word_count+=1


    percentage_complex_words = complex_word_count / word_count 

    average_sentence_length = word_count / sentence_count
    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)


    return {
        'AVG SENTENCE LENGTH': average_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
        'FOG INDEX': fog_index,
        'COMPLEX WORD COUNT':complex_word_count
            }

# Average Number of Words Per Sentence
def average_words_per_sentence(text):
    sentences = sent_tokenize(text)
    text = re.sub(r'[^\w\s.]', '', text)
    text = text.lower()
    words = word_tokenize(text)

    avg_words = len(words) / len(sentences)
    return {'AVG NUMBER OF WORDS PER SENTENCE':avg_words}

# Word Count
def cleaned_words(text):
    # Tokenize the text into words
    text = re.sub(r'[^\w\s.]', '', text)
    words = nltk.word_tokenize(text.lower())

    # Get stopwords and punctuation set
    punctuation = set(string.punctuation)

    # Remove stopwords and punctuation, and count cleaned words
    cleaned_words = [word for word in words if word.lower() not in stop_words and word.lower() not in punctuation]
    return {'WORD COUNT':len(cleaned_words)}



def personal_pronoun_count(text):
    # Define the personal pronouns to search for
    pronouns = ["I", "we", "my", "ours", "us"]

    # Regex pattern to find the personal pronouns
    pattern = r'\b(?:' + '|'.join(pronouns) + r')\b'

    # Use regex to find all matches in the text
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    # Filter out occurrences of "US" which is not a personal pronoun
    filtered_matches = 0
    for word in matches:
        if word.lower() not in ["us"]:
            filtered_matches += 1

    return {'PERSONAL PRONOUNS':filtered_matches}

def average_word_length(text):
    text = re.sub(r'[^\w\s.]', '', text)
    words = word_tokenize(text.lower())
    total_character=0
    for word in words:
        total_character+=len(word)
    return {'AVERAGE WORD LENGTH':total_character/len(words)}

def process_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        sentiment_result = sentimental_analysis(text)
        readability_result = readability_analysis(text)

        result = {
            'filename': os.path.splitext(os.path.basename(filepath))[0],
            **sentiment_result,
            **readability_result,
            **average_words_per_sentence(text),
            **cleaned_words(text),
            **personal_pronoun_count(text),
            **average_word_length(text)
        }
        return result
def process_text_files(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            result = process_text_file(filepath)
            results.append(result)
    return results

# Example usage
directory_path = 'extracted_texts'
results = process_text_files(directory_path)


# input excel file
df = pd.read_excel('input.xlsx')
# new df with output
df1 = pd.DataFrame(results)

# change column name to join datframe
df1.rename(columns={'filename': 'URL_ID'}, inplace=True)
# print(df1)

# two dataframes joined
merged_df = pd.merge(df, df1, on='URL_ID', how='outer')

# save the dataframe in csv file
merged_df.to_csv('Output_Data.csv')
