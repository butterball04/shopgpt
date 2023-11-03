# ShopGPT

ShopGPT provides large knowledge context using pinecone vector database and openai embeddings-api to help GPT provide product recommendations for your e-commerce store.

## Installation

1. Clone the repository.
2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your `.env` file with your OpenAI and Pinecone credentials.

## How to Use

### Creating Embeddings with `train.py`

To generate and upload embeddings to Pinecone:

1. Prepare a CSV file with product data.
2. Run:

   ```bash
   python train.py --file <path_to_your_csv>
   ```

### Running `gpt.py` to Query

To start querying with GPT:

```bash
python gpt.py
```

Follow the prompts to input your questions and receive answers based on the embeddings.

## Support

For issues and questions, open an issue on the GitHub repository.

## Contributing

Contributions are welcome. Please fork the repo and submit a pull request.

## License

Licensed under the MIT License.
