# CodeAlpha_Translationtool
import string
import nltk

# This line MUST be right after importing nltk.
# It tells your script the exact address of the data.
nltk.data.path.append("C:/nltk_data")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class FAQChatbot:
    """A simple FAQ chatbot that uses TF-IDF and cosine similarity."""

    def __init__(self, faq_data):
        self.faq_data = faq_data
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.faq_questions = list(self.faq_data.keys())
        
        processed_questions = [self._preprocess(q) for q in self.faq_questions]
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(processed_questions)

    def _preprocess(self, text):
        text = text.lower()
        tokens = word_tokenize(text)
        processed_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in string.punctuation and word not in self.stop_words
        ]
        return " ".join(processed_tokens)

    def get_response(self, user_query):
        processed_query = self._preprocess(user_query)
        query_vector = self.vectorizer.transform([processed_query])
        
        similarities = cosine_similarity(query_vector, self.question_vectors)
        most_similar_idx = similarities.argmax()
        
        confidence_threshold = 0.2
        if similarities[0, most_similar_idx] < confidence_threshold:
            return "I'm sorry, I don't have an answer for that. Please try asking in a different way."
        else:
            best_matching_question = self.faq_questions[most_similar_idx]
            return self.faq_data[best_matching_question]

def main():
    faq_knowledge_base = {
        "What are the system requirements?": "Our product runs on Windows 10/11 and macOS 12.0 or newer.",
        "How do I reset my password?": "You can reset your password by clicking the 'Forgot Password' link on the login page.",
        "Do you offer a free trial?": "Yes, we offer a 14-day free trial with full access to all features.",
        "What is your refund policy?": "We have a 30-day money-back guarantee. If you are not satisfied, contact support for a full refund.",
        "How can I contact customer support?": "You can reach our support team 24/7 via email at support@example.com or through our live chat."
    }
    
    chatbot = FAQChatbot(faq_knowledge_base)
    
    print("Hello! I'm a FAQ chatbot. Ask me anything about our product or type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        answer = chatbot.get_response(user_input)
        print(f"Chatbot: {answer}")

if __name__ == "__main__":
    main()
