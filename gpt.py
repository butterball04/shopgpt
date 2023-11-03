import openai
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')


def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def display_with_sources(response_content, matches):
    response = f"ShopGPT: {response_content}\n\nSources:\n"
    for match in matches:
        title = match['metadata']['title']
        description = match['metadata']['description']
        tags = match['metadata']['tags']
        price = match['metadata']['price']
        response += f"- [{title}]|{description}|{tags}|{price})\n"
    print(response)


messages = []


def query():
    while True:
        # Define index name
        index_name = 'shopgpt'
        embed_query = 'text-embedding-ada-002'
        system_msg = 'You are a helpful shopping assistant for a shopify website, please help the user to make the right purchase. Don\'t suggest any products that are not given in the prompt context.'

        # Initialize connection to Pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

        # Connect to the index and view index stats
        index = pinecone.Index(index_name)

        user_message = input("You: ")
        if user_message.lower() == "quit":
            break

        # enrich prompt by fetching tags from chatgpt
        enrich_message = 'Given a prompt create list of tags for the categories mentioned to enrich the prompt and help with better product recommendations. Don\'t include tags that you can\'t predict with enough accuracy.Categories: Occasion Style, Embellishments, Season, Material, Prompt:' + user_message
        enrich_messages = []
        enrich_messages.append(
            {"role": "assistant", "content": enrich_message})
        enrich_chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=enrich_messages
        )
        enrich_tags = enrich_chat['choices'][0]['message']['content']

        # Perform the search based on the user's query and retrieve the relevant sources
        embed_input = user_message + "\n" + enrich_tags + "\n" + str(messages)
        embed_query = openai.Embedding.create(
            input=[embed_input],
            engine=embed_query
        )

        # retrieve from Pinecone
        query_embeds = embed_query['data'][0]['embedding']

        # get relevant contexts (including the questions)
        response = index.query(query_embeds, top_k=5, include_metadata=True)
        matches = response['matches']

        # get list of retrieved text
        contexts = [item['metadata']['title'] + item['metadata']['description'] +
                    str(item['metadata']['price']) + item['metadata']['tags'] for item in matches]

        # concatenate contexts and user message to generate augmented query
        augmented_query = " --- ".join(contexts) + " --- " + user_message
        messages.append({"role": "user", "content": augmented_query})
        messages.append({"role": "system", "content": system_msg})

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        assistant_message = chat['choices'][0]['message']['content']
        messages.append({"role": "assistant", "content": assistant_message})
        display_with_sources(assistant_message, matches)


query()
