import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import difflib
import joblib

# Load the dataset with correct names
df = pd.read_csv(r"D:\museum_chatbot\dataset\museums.csv", header=None, names=['correct_name'], encoding='latin-1')

# Generate synthetic variations by removing spaces, adding common misspellings, and extra spaces
def generate_variations(name):
    variations = [name.replace(' ', '').lower(), name.lower(), name.replace(' ', '  ').lower()]
    return variations

# Create a new DataFrame with variations
data = {'correct_name': [], 'variation': []}
for name in df['correct_name']:
    for variation in generate_variations(name):
        data['correct_name'].append(name)
        data['variation'].append(variation)

df_variations = pd.DataFrame(data)

# Preprocess text: lowercase and remove extra spaces
def preprocess(text):
    return ' '.join(text.lower().split())

df_variations['correct_name'] = df_variations['correct_name'].apply(preprocess)
df_variations['variation'] = df_variations['variation'].apply(preprocess)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df_variations, test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer().fit(train_df['correct_name'])
train_vectors = vectorizer.transform(train_df['correct_name'])
test_vectors = vectorizer.transform(test_df['variation'])

# Function to find the best match using cosine similarity and difflib
def find_best_match(query, vectorizer, train_vectors, train_names, threshold=0.5):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, train_vectors)
    best_match_index = similarities.argmax()
    best_match_name = train_names[best_match_index]
    
    # Use difflib to refine the best match
    difflib_scores = [difflib.SequenceMatcher(None, query, name).ratio() for name in train_names]
    best_difflib_match_index = difflib_scores.index(max(difflib_scores))
    best_difflib_match_name = train_names[best_difflib_match_index]
    
    # Combine scores
    combined_scores = [(similarity + difflib_score) / 2 for similarity, difflib_score in zip(similarities[0], difflib_scores)]
    best_combined_match_index = combined_scores.index(max(combined_scores))
    best_combined_match_name = train_names[best_combined_match_index]
    
    # Check if the best match score is above the threshold
    if combined_scores[best_combined_match_index] < threshold:
        return None
    
    return best_combined_match_name

# Test the model
correct_predictions = 0

for _, row in test_df.iterrows():
    best_match = find_best_match(row['variation'], vectorizer, train_vectors, train_df['correct_name'].values)
    
    # ✅ Handle cases where best_match is None
    if best_match is not None and best_match == row['correct_name']:
        correct_predictions += 1

# ✅ Avoid division by zero
accuracy = correct_predictions / len(test_df) if len(test_df) > 0 else 0

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Example usage with a sample query
query = "amaravathi archeological"
best_match = find_best_match(query, vectorizer, train_vectors, train_df['correct_name'].values)
print(f"Best match for '{query}': {best_match}")

# Save the model components
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(train_df['correct_name'].values, 'train_names.pkl')
joblib.dump(train_vectors, 'train_vectors.pkl')

print("Model components saved successfully.")