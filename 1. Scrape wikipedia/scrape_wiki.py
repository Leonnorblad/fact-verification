# Function used to scrape random wikipedia articles
import wikipedia
import pandas as pd
import tqdm
import tiktoken
import spacy
nlp = spacy.load('en_core_web_sm')
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def check_texts(text):
    """
    Takes a list of texts and returns False if one of the texts is 'nan' or not a string.
    """
    if not (isinstance(text, str) and text.lower() != 'nan'):
        return False
    return True

def scrape_wiki(title):
    """
    Downloads the content of a single wikipedia article.
    First tries to scrape 'title', then samples a new article title
    if the page is not available.
    """
    try:
        page = wikipedia.page(title, auto_suggest=False)
        article = page.content
        collected_title = page.title
        if check_texts(article):
            return collected_title, article
        else:
            new_title = wikipedia.random()
            return scrape_wiki(new_title)
    except:
        new_title = wikipedia.random()
        return scrape_wiki(new_title)

def scrape_data(n):
    """
    Randomly downloads n articles.

    Returns a dataframe with n rows and two columns:
        Title: Title of the article
        Article: The full wikipedia article
    """
    titles = []
    articles = []

    for i in tqdm.tqdm(range(n)):
        title = wikipedia.random()
        title, article = scrape_wiki(title)

        titles.append(title)
        articles.append(article)
    return_df = pd.DataFrame({"Title":titles,
                             "Article":articles})
    return return_df

def count_tokens(text):
    """
    Returns the number of tokens (gpt 3.5 turbo) in the given text.
    """
    return len(encoding.encode(text))

def short_summary(text, max_tokens=100, min_tokens=20):
    """
    Creates a summary of 'text' containing whole sentences
    shorter than 'max_tokens' and longer than 'min_tokens'.
    
    The function works by adding sentences to the summary
    until the maximum token length is reached.

    Returns nan if the summary could not be created.
    This happends if the text is shorter than 'min_tokens'
    or the first sentence is longer than 'max_tokens'.
    """
    # If the article is to short -> do not use
    tokens_text = count_tokens(text)
    if tokens_text <= min_tokens:
        return float('nan')
    
    # Slplit text into snetences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # If the first sentence is to long -> do not use
    tokens_first_sent = count_tokens(sentences[0])
    if tokens_first_sent>=max_tokens:
        return float('nan')
    
    # Start construct output text with the first sentence
    output_text = sentences[0]
    for i in range(1, len(sentences)):
        sentence = sentences[i]
        # Try to add one more sentence
        next_output = output_text + " " + sentence
        next_num_tokens = count_tokens(next_output)

        if next_num_tokens <= max_tokens:
            output_text = next_output
        else:
            break
    # Check that the collected summary is not to short
    len_output = count_tokens(output_text)
    if len_output<=min_tokens:
        return float('nan')
    return output_text