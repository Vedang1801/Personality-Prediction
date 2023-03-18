import re
import time
import nltk
import nltk_download_utils
import pickle
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the file paths
MODEL_FILE_PATH = 'Models/model_cat.pkl'

st.set_page_config(page_title="Personality-Prediction",layout="centered",  page_icon="🧠")
# Define the Streamlit app


def app():
    # Add a title to the app
    st.title("Personality Prediction Questionnaire")
    st.write("-----------------------------------------------------")

    # Define the questions and options
    questions = [
        "Question 1: I enjoy being in large groups of people.",
        "Question 2: I feel energized after interacting with many people.",
        "Question 3: I find it easy to approach and talk to strangers.",
        "Question 4: I like being the center of attention in social situations.",
        "Question 5: I prefer spending time alone over being with others.",
        "Question 6: I feel drained after spending a lot of time around people and need alone time to recharge.",
        "Question 7: I prefer to work independently and avoid group projects if possible.	",
        "Question 8: I prefer to listen and observe in social situations, rather than being the one to initiate conversation.",
        "Question 9: I prefer to focus on the facts and concrete information when making decisions.",
        "Question 10: I am more concerned with the present moment and practical realities than future possibilities.",
        "Question 11: I rely on past experiences to guide my decisions and actions.",
        "Question 12: I often see patterns and connections in things that others don't.",
        "Question 13: I trust my instincts and rely on my gut feeling when making decisions.",
        "Question 14: I am highly imaginative and enjoy exploring new ideas and possibilities.",
        "Question 15: I am more interested in the theoretical and abstract aspects of a situation.",
        "Question 16: I like to have a detailed plan before taking action.",
        "Question 17: We should make decisions based on logical reasoning and facts.",
        "Question 18: The best decision is the one that maximizes efficiency and productivity.",
        "Question 19: Objectivity should be the primary consideration when making decisions.",
        "Question 20: The most important factor in decision making is the analysis of available data.",
        "Question 21: I believe that we should prioritize our personal values in decision making.",
        "Question 22: I think we should consider how our actions will affect others before making a decision.",
        "Question 23: I feel that relationships and emotional connections should play a role in decision making.",
        "Question 24: I believe that intuition should guide our decisions in certain situations.",
        "Question 25: I like to have a clear plan and stick to it.",
        "Question 26: I enjoy having a structured and organized approach to tasks",
        "Question 27: I feel most comfortable when I know what to expect in a situation.",
        "Question 28: I believe in making decisions and sticking to them.",
        "Question 29: I prefer to stay flexible and adapt to changing situations.",
        "Question 30: I like to keep my options open and explore different possibilities.",
        "Question 31: I enjoy going with the flow and being spontaneous.",
        "Question 32: I think it's important to remain open to new ideas and possibilities."
    ]

    options = ['Strongly Disagree', 'Disagree',
               'Neutral', 'Agree', 'Strongly Agree']

    text_questions = [
        {'question': 'Question 1: How do you prefer to spend your free time?', 'options': ["I prefer to spend my free time alone, reading a book or watching a movie. I find it relaxing and rejuvenating to be in my own space and not have to socialize. I recharge my batteries by being alone.",
                                                                                           "I prefer to spend my free time with others, going out with friends or trying new activities. I find it energizing and fun to be around people and have new experiences. I recharge my batteries by being with others."]},
        {'question': 'Question 2: How do you prefer to approach problem-solving?', 'options': ["I prefer to approach problem-solving by using my practical knowledge and experience. I like to focus on the concrete facts and details and come up with a solution that is based on tangible evidence. I find it more satisfying when I can find a solution that is tried and true.",
                                                                                               "I prefer to approach problem-solving by using my imagination and creativity. I like to focus on the big picture and the possibilities, and come up with a solution that is unique and innovative. I find it more satisfying when I can find a solution that is novel and unexpected."]},
        {'question': 'Question 3: When making important decisions, how do you prioritize your values and beliefs?', 'options': ["When making important decisions, I prioritize logic and objectivity. I try to consider the consequences of my actions objectively and make decisions based on what is fair and just. I find it more important to maintain a logical and consistent decision-making process, rather than basing my decisions on my emotions or personal values.",
                                                                                                                                "When making important decisions, I prioritize my personal values and beliefs. I try to consider the impact of my actions on the people involved and make decisions based on what is in line with my heart. I find it more important to maintain a consistent set of values and beliefs, rather than basing my decisions on pure logic."]},
        {'question': 'Question 4: How do you typically approach your daily schedule and tasks?', 'options': ["I prefer to have a structured and organized approach to my daily tasks. I like to have a plan for my day and follow it as closely as possible. I feel more comfortable when I have completed my tasks and my schedule is in order. I find that having a structured routine helps me to be more productive and achieve my goals.",
                                                                                                             "I prefer a flexible and spontaneous approach to my daily tasks. I like to leave room for surprises and unexpected opportunities. I feel more comfortable when I have the freedom to change my plans if something more interesting or exciting comes up. I find that having a flexible and spontaneous approach helps me to be more adaptable and go with the flow."]}
    ]

    # Collect input from the user
    responses = collect_responses(questions, options, text_questions)
    st.write(" ")
    st.write("-----------------------------------------------------")
    st.header("Personality Type Predictor")
    st.caption("Click On The Predict Button to know your Personality")

    # Make a prediction and display the result
    st.write(" ")
    if st.button('PREDICT :thought_balloon:'):
        try:
            # Load the model
            loaded_model = load_model(MODEL_FILE_PATH)

            # Convert input data to a format that can be used by the model
            x = preprocess_input(responses)

            # Make a prediction using the model
            y_pred = loaded_model.predict(x)

            # Convert the predicted class to an integer value
            predicted_class = int(y_pred)

            # Define a switch case function
            def switch_case(y):
                switcher = {
                    0: "ENFJ(The Teacher): Warm and empathetic, ENFJs are natural leaders who bring out the best in others. They are excellent communicators and know how to bring harmony to a group. ENFJs are driven by a sense of purpose and a desire to make a positive impact on the world.",
                    1: "ENFP(The Champion): Enthusiastic and optimistic, ENFPs are full of ideas and always ready for an adventure. They are excellent at understanding and connecting with others, and enjoy bringing people together. ENFPs are highly creative and often excel at finding new and innovative solutions to problems.",
                    2: "ENTJ(The Commander): Strategic and driven, ENTJs are natural leaders who excel at creating and executing plans. They are confident and assertive, and are not afraid to make tough decisions. ENTJs are highly ambitious and always looking for new challenges to conquer.",
                    3: "ENTP(The Visionary): Quick-witted and always curious, ENTPs are natural problem-solvers who enjoy exploring new ideas and possibilities. They are enthusiastic and energetic, and are always looking for ways to shake things up. ENTPs are highly independent and value their freedom to think and act as they please.",
                    4: "ESFJ(The Provider): Warm and supportive, ESFJs are natural caregivers who enjoy helping others. They are highly organized and reliable, and value traditions and stability. ESFJs are driven by a strong sense of responsibility and a desire to make a positive impact on those around them.",
                    5: "ESFP(The Performer: Fun-loving and spontaneous, ESFPs are the life of the party and enjoy being in the spotlight. They are highly social and enjoy making connections with others. ESFPs live in the moment and value their freedom to enjoy life to the fullest.",
                    6: "ESTJ(The Supervisor): Practical and down-to-earth, ESTJs are natural leaders who are highly organized and efficient. They value tradition and order, and are driven by a sense of duty and responsibility. ESTJs are highly assertive and confident, and are not afraid to take charge in a crisis.",
                    7: "ESTP(The Dynamo): Bold and daring, ESTPs are natural risk-takers who enjoy living life on the edge. They are highly action-oriented and enjoy taking control of a situation. ESTPs value their independence and enjoy exploring new experiences and opportunities.",
                    8: "INFJ(The Counselor): Intuitive and empathetic, INFJs are natural healers who enjoy helping others. They are highly imaginative and often have a strong sense of their life's purpose. INFJs are driven by a strong sense of morality and a desire to make a positive impact on the world.",
                    9: "INFP(The Idealist): Idealistic and creative, INFPs are natural dreamers who enjoy exploring their own thoughts and feelings. They are highly empathetic and value deep connections with others. INFPs are driven by a strong sense of purpose and a desire to make a positive impact on the world.",
                    10: "INTJ(The Architect): Strategic and visionary, INTJs are natural leaders who excel at developing and executing plans. They are highly analytical and enjoy finding new and innovative solutions to problems. INTJs are driven by a strong sense of purpose and a desire to make a positive impact on the world.",
                    11: "INTP(The Architect): They are creative, innovative, and logical problem-solvers. They enjoy exploring new ideas and theories, and are not afraid to challenge traditional thinking.",
                    12: "ISFJ(The Protector): They are responsible, reliable, and nurturing individuals. They have a strong sense of duty and put the needs of others before their own. They value tradition and stability.",
                    13: "ISFP(The Composer): They are sensitive, imaginative, and spontaneous individuals. They are often artistic, and enjoy living in the moment. They value their independence and freedom.",
                    14: "ISTJ(The Inspector): They are practical, dependable, and methodical individuals. They value order, stability, and structure, and are committed to their responsibilities. They are often the backbone of organizations.",
                    15: "ISTP(The Craftsman): They are hands-on individuals who enjoy taking things apart and figuring out how they work. They are confident, independent, and resourceful, and value action and results over theoretical ideas."
                }
                return switcher.get(y, "Invalid class")

     # Display a progress bar while the model is making the prediction
            progress_bar = st.progress(0)
            progress_text = st.empty()
            for i in range(100):
                time.sleep(0.03)
                progress_bar.progress(i + 1)
                progress_text.text("Hold On Making prediction... {}%".format(i + 1))
            progress_bar.empty()
            progress_text.empty()

        # Make a prediction using the model
            y_pred = loaded_model.predict(x)

        # Convert the predicted class to an integer value
            predicted_class = int(y_pred)

        # Call the switch case function
            result = switch_case(predicted_class)

        # Display the prediction to the user
            st.success("Your personality type is {}".format(result), icon='✅')
        # Display MBTI Accordian
            mbti_types = {
    "I" : "  INTROVERSION  ",
    "E" : "  EXTROVERSION  ",
    "S" : "  SENSING  ",
    "N" : "  INTUTION  ",
    "T" : "  THINKING  ",
    "F" : "  FEELING  ",
    "J" : "  JUDGING  ",
    "P" : "  PERCIEVING  ",

    }
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.caption("Additional Information")
            with st.expander("Click here to see the MBTI types and their full forms"):
                for mbti_type, full_form in mbti_types.items():
                    st.write(f"{mbti_type}: {full_form}")
            
        except Exception as e:
            st.error('Please provide a response to the textual question.')


def collect_responses(questions, options, text_questions):
    st.subheader("MCQ Questions")
    st.caption("Select an option for each question.")
    # Collect input for numerical questions with 5 predefined options
    responses = []
    for i in range(len(questions)):
        response = st.selectbox(questions[i], options)
        index_of_response = options.index(response) + 1
        responses.append(index_of_response)

    st.write(" ")
    st.write("-----------------------------------------------------")
    st.header("Textual Questions: ")
    st.caption("Select an option for each question.")
    # Collect input for textual questions with different predefined options
    text_responses = []
    for i in range(len(text_questions)):
        response = st.selectbox(
            text_questions[i]['question'], text_questions[i]['options'])
        text_responses.append(response)

    # Collect input for the fifth textual question using text_input
    text_response_string = ''
    
    text_response = st.text_input('Describe a challenging situation you have faced in the past and how you overcame it.',max_chars=600,placeholder="Enter Answer Here")
    if text_response:
        text_response_string = text_response

    # Combine all the textual responses into a single string
    text_response_string = ' '.join(text_responses) + ' ' + text_response_string

    # Remove URLs using regular expressions
    text_response_string = re.sub(r"http\S+", "", text_response_string)

    # Remove special characters using regular expressions
    text_response_string = re.sub(r"[^a-zA-Z0-9]+", " ", text_response_string)

    # Tokenize the text_response_string into words
    words = nltk.word_tokenize(text_response_string)

    # Remove stop words using NLTK's English stop words list
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.lower() not in stop_words]

    # Join the words back into a string
    text_response_string = " ".join(words)

    # Initialize the TF-IDF Vectorizer with 32 features
    tfidf = TfidfVectorizer(max_features=32)

    # Fit and transform the textual data
    result_tfidf = tfidf.fit_transform([text_response_string]).toarray()

    # Append the textual questions answer to numerical questions answer
    responses.extend(result_tfidf.tolist())

    # Extend the last list in the array to the existing list
    responses.extend(responses.pop())

    # Convert the user's answers list to a numpy array
    responses = np.array(responses).reshape(1, -1)

    return responses


def load_model(file_path):
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model


def preprocess_input(responses):
    return np.array(responses).reshape(1, -1)


if __name__ == '__main__':
    app()
