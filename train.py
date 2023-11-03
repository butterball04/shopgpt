import re
import openai
import pinecone
import csv
import pandas as pd
from uuid import uuid4
from tqdm.auto import tqdm
import time
from dotenv import load_dotenv
import os
import argparse

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')

HTML_CLEANER = re.compile('<.*?>')


def clean_text(serie):
    serie = serie.replace('\n', ' ')
    serie = serie.replace('\\n', ' ')
    serie = serie.replace('  ', ' ')
    serie = serie.replace('  ', ' ')
    serie = re.sub(HTML_CLEANER, '', serie)

    return serie


def read_clean_csv(input_file):
    output_file = input_file.replace('.csv', '_cleaned.csv')

    with open(input_file, newline='') as infile, open(output_file, 'w', newline='') as outfile:
        # Create a DictReader for the input file and DictWriter for the output file
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        # Iterate over each row, clean the description field, and write the updated row to the output file
        for row in reader:
            row['title'] = clean_text(row['title'])
            row['description'] = clean_text(row['description'])
            row['tags'] = clean_text(row['tags'])
            writer.writerow(row)


def train(input_file):
    # Clean the data
    read_clean_csv(input_file)
    output_file = input_file.replace('.csv', '_cleaned.csv')

    # Read the CSV file into a dataframe
    maindf = pd.read_csv(output_file)
    df = maindf.iloc[:1000]

    # Add an 'id' column to the DataFrame
    df['id'] = [str(uuid4()) for _ in range(len(df))]

    # Display the first 5 rows of the updated dataframe
    print(df.head())

    # Define index name
    index_name = 'shopgpt'

    # Initialize connection to Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

    # Check if index already exists, create it if it doesn't
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536, metric='cosine')

    # Connect to the index and view index stats
    index = pinecone.Index(index_name)
    print(index.describe_index_stats())

    batch_size = 100  # how many embeddings we create and insert at once

    # Convert the DataFrame to a list of dictionaries
    new_data = df.to_dict(orient='records')

    for i in tqdm(range(0, len(new_data), batch_size)):
        # find end of batch
        i_end = min(len(new_data), i+batch_size)
        meta_batch = new_data[i:i_end]
        # get ids
        ids_batch = [x['id'] for x in meta_batch]
        # get texts to encode
        texts = [x['title'] + x['description'] + x['tags'] for x in meta_batch]
        # create embeddings (try-except added to avoid RateLimitError)
        try:
            res = openai.Embedding.create(
                input=texts, engine="text-embedding-ada-002")
        except:
            done = False
            while not done:
                time.sleep(5)
                try:
                    res = openai.Embedding.create(
                        input=texts, engine="text-embedding-ada-002")
                    done = True
                except:
                    pass
        embeds = [record['embedding'] for record in res['data']]
        # cleanup metadata
        meta_batch = [{
            'url': x['url'],
            'title': x['title'],
            'description': x['description'],
            'price': x['price'],
            'tags': x['tags']
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        index.upsert(vectors=to_upsert)

    index.describe_index_stats()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create embeddings for your data using a CSV file')
    parser.add_argument('--file', required=True,
                        help='The path to the CSV file to create embeddings.')

    args = parser.parse_args()
    train(args.file)
