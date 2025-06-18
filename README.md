#  Natural Language to SQL Translator

This project translates natural language queries (e.g., "show me all users") into SQL commands using a Sequence-to-Sequence (Seq2Seq) model with GRU units in TensorFlow. It’s a simple yet powerful example of how deep learning can be used for natural language interfaces to databases.

## 🔍 Demo

**Input:**  
`show me all products`

**Output:**  
```sql
SELECT * FROM products ;
````

## 🧠 How It Works

### Seq2Seq Architecture

The model uses a **Sequence-to-Sequence (Seq2Seq)** architecture, which consists of:

* **Encoder**: Takes in the input sentence and encodes it into a fixed-size context vector using embeddings and a GRU layer.
* **Decoder**: Takes the context vector and generates the SQL command word-by-word using another GRU.

This architecture is commonly used for tasks like machine translation, chatbot responses, and more.

### Greedy Search Decoding

At inference time, we use **greedy decoding**, which selects the token with the highest probability at each step instead of exploring all possible sequences. While not optimal compared to beam search, it's simple and works well for small datasets like ours.

## 🛠 Setup

1. Clone the repo

   ```bash
   git clone https://github.com/yourusername/nlp2sql.git
   cd nlP_to_sql
   ```

2. Install dependencies

   ```bash
   pip install tensorflow numpy
   ```

3. Run training and test

   ```bash
   python nlp2sql.py
   ```

## 📦 Files

* `nlp2sql.py` – Main file with training and inference code

## 🚀 Future Improvements

* Replace greedy decoding with beam search
* Train on larger and more diverse NL-SQL datasets
* Add web interface using Streamlit


