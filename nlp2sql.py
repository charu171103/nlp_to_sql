import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense
import numpy as np

# ==== Data ====
input_texts = [
    "show me all users",
    "show me all products",
    "show me all orders",
    "get users with age greater than 30",
    "get all orders with price greater than 500",
    "count number of users",
    "count number of orders",
    "get all product names",
    "show all users with email verified",
    "show products with stock less than 50",
    "list users from USA",
    "find products with rating greater than 4",
    "count products in category electronics",
    "get total revenue",
    "show top 10 products by sales",
    "get orders placed today",
    "get users registered this month",
    "show last 5 orders",
    "list orders with status pending",
    "find users who spent more than 1000"
]

target_texts = [
    "SELECT * FROM users ;",
    "SELECT * FROM products ;",
    "SELECT * FROM orders ;",
    "SELECT * FROM users WHERE age > 30 ;",
    "SELECT * FROM orders WHERE price > 500 ;",
    "SELECT COUNT ( * ) FROM users ;",
    "SELECT COUNT ( * ) FROM orders ;",
    "SELECT name FROM products ;",
    "SELECT * FROM users WHERE email_verified = true ;",
    "SELECT * FROM products WHERE stock < 50 ;",
    "SELECT * FROM users WHERE country = 'USA' ;",
    "SELECT * FROM products WHERE rating > 4 ;",
    "SELECT COUNT ( * ) FROM products WHERE category = 'electronics' ;",
    "SELECT SUM ( total_price ) FROM orders ;",
    "SELECT * FROM products ORDER BY sales DESC LIMIT 10 ;",
    "SELECT * FROM orders WHERE order_date = CURRENT_DATE ;",
    "SELECT * FROM users WHERE registration_date >= DATE_TRUNC ( 'month', CURRENT_DATE ) ;",
    "SELECT * FROM orders ORDER BY order_date DESC LIMIT 5 ;",
    "SELECT * FROM orders WHERE status = 'pending' ;",
    "SELECT * FROM users WHERE total_spent > 1000 ;"
]

# ==== Tokenizer ====
input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False)
target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False)


# Add special tokens <SOS> and <EOS>
target_texts = ["<SOS> " + t + " <EOS>" for t in target_texts]

# Fit tokenizers
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

input_tensor = input_tokenizer.texts_to_sequences(input_texts)
target_tensor = target_tokenizer.texts_to_sequences(target_texts)

# Pad sequences
input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, padding='post')
target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, padding='post')

# Vocab sizes
INPUT_DIM = len(input_tokenizer.word_index) + 1
OUTPUT_DIM = len(target_tokenizer.word_index) + 1
print(f"Input Vocab Size: {INPUT_DIM}, Target Vocab Size: {OUTPUT_DIM}")

# ==== Hyperparameters ====
HID_DIM = 128
BATCH_SIZE = 8
EPOCHS = 300

# ==== Encoder ====
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, hid_dim):
        super().__init__()
        self.embedding = Embedding(vocab_size, hid_dim)
        self.gru = GRU(hid_dim, return_state=True)
    
    def call(self, x):
        x = self.embedding(x)
        output, state = self.gru(x)
        return state

# ==== Decoder ====
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, hid_dim):
        super().__init__()
        self.embedding = Embedding(vocab_size, hid_dim)
        self.gru = GRU(hid_dim, return_state=True)
        self.fc = Dense(vocab_size)
    
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        x = self.fc(output)
        return x, state

# ==== Instantiate ====
encoder = Encoder(INPUT_DIM, HID_DIM)
decoder = Decoder(OUTPUT_DIM, HID_DIM)

# ==== Optimizer + Loss ====
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ==== Training Step ====
@tf.function
def train_step(inp, targ):
    loss = 0
    with tf.GradientTape() as tape:
        hidden = encoder(inp)
        dec_input = tf.expand_dims([target_tokenizer.word_index['<SOS>']] * targ.shape[0], 1)
        
        for t in range(1, targ.shape[1]):
            predictions, hidden = decoder(dec_input, hidden)
            loss += loss_object(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
    
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss / targ.shape[1]

# ==== Training Loop ====
dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(len(input_tensor))
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

for epoch in range(EPOCHS):
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(dataset):
        batch_loss = train_step(inp, targ)
        total_loss += batch_loss
    if (epoch+1) % 50 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss.numpy() / batch:.4f}')

# ==== Inference (Greedy Decoding) ====
def translate(sentence, max_len=40):
    sentence_seq = input_tokenizer.texts_to_sequences([sentence])
    sentence_seq = tf.keras.preprocessing.sequence.pad_sequences(sentence_seq, maxlen=input_tensor.shape[1], padding='post')
    
    hidden = encoder(tf.convert_to_tensor(sentence_seq))
    
    dec_input = tf.expand_dims([target_tokenizer.word_index['<SOS>']], 0)
    result = []
    
    for _ in range(max_len):
        predictions, hidden = decoder(dec_input, hidden)
        predicted_id = tf.argmax(predictions[0]).numpy()

        
        if predicted_id == target_tokenizer.word_index['<EOS>']:
            break
        
        word = target_tokenizer.index_word.get(predicted_id, '')
        result.append(word)
        dec_input = tf.expand_dims([predicted_id], 0)
    
    return ' '.join(result)

# ==== Test ====
if __name__ == "__main__":
    # Training loop...
    for epoch in range(EPOCHS):
        batch_loss = train_step(inp, targ)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {batch_loss.numpy():.4f}")
    
    # Ask user for input after training
    test_sentence = input("Enter your question: ")
    sql_output = translate(test_sentence)
    print("Generated SQL:", sql_output)