# Functions used to call the openai API
from tqdm import tqdm
import pickle
from openai import OpenAI
client = OpenAI()


def save_lst(obj, name):
    """
    Saves a list 'obj' with filename 'name'
    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

def preprocess_articles(scraped_articles, num_samples):
    """
    Takes 'scraped_articles' and devides it in two.
    Used to devide articles into two groups,
    one for true statements and one for false. 

    Arguments
        scraped_articles: pandas DataFrame with scraped articles
        num_samples: Number of samples to use

    Returns:
        scraped_summary0: first part
        scraped_summary1: second part
    """
    # Extract all samples up to 'num_samples'
    scraped_articles = scraped_articles.iloc[:num_samples]
    # Find indec of middle
    split_index = len(scraped_articles)//2
    # Slice for the first part
    scraped_summary0 = scraped_articles.iloc[:split_index]
    # Slce for the second part
    scraped_summary1 = scraped_articles.iloc[split_index:]
    return scraped_summary0, scraped_summary1

def construct_prompt_true(article_name, article):
    """
    Creates a promt to generate a true statement.

    Arguments
        article_name: Name of the article.
        article: Content (text) of the article.
    
    Returns
        Prompt.
    """
    return [
        {"role": "system", "content": "Answers must be written in a way that can be understood without context. Your response must consist of a single sentence shorter than 8 words."},
        {"role": "user", "content": f"Pick one single detail in this text about {article_name} and write it as a standalone statement: '{article}'"}
        ]

def construct_prompt_false(article_name, article):
    """
    Creates a promt to generate a false statement.

    Arguments
        article_name: Name of the article.
        article: Content (text) of the article.
    
    Returns
        Prompt.
    """
    return [
        {"role": "system", "content": "Answers must be written in a way that can be understood without context. Your response must consist of a single sentence shorter than 8 words and be a false statement."},
        {"role": "user", "content": f"Change one single detail in this text about {article_name} and write it as a standalone false statement: '{article}'"}
        ]

def generate_statements(summary_data, prompt_fun, statement_type):
    """
    Send promt to gpt-3.5-turbo and saves the response

    Arguments:
        summary_data: pd.DataFrame with article title, and summary.
        prompt_fun: Function to use to generate Prompt.
        statement_type: Binary. 1 for true statement. 0 for false statement.
    
    Returns:
        full_report: Full report (response) from the API.
        summary_data: pd.DataFrame with the input data and two new columns:
            Statement: The generated statement
            Label: 
    """
    full_report = []
    generated_statements = []
    for i, row in tqdm(summary_data.iterrows()):
        prompt = prompt_fun(row["Title"], row["Summary"])

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            max_tokens=20
            )
        
        full_report.append(completion)
        response = completion.choices[0].message.content
        generated_statements.append(response)
    summary_data["Statement"] = generated_statements
    summary_data["Label"] = statement_type

    return full_report, summary_data