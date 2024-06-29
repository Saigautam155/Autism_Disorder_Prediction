import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
import hashlib
import sqlite3
from PIL import Image
from streamlit_option_menu import option_menu
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(page_title="Autism Chatbot", layout="wide")

# Load environment variables from .env file
load_dotenv()

# Configure Generative AI
genai.configure(api_key="Give Your Google Api Code")
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Database functions
conn = sqlite3.connect('data.db')
c = conn.cursor()


def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')


def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username, password) VALUES (?, ?)', (username, password))
    conn.commit()


def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    return data


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


# Streamlit App
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Login/Signup", "Home", "ASD Assessment", "Statistics", "Q&A Chatbot", "Contact Us"],
        icons=["person", "house", "clipboard-check", "bar-chart", "chat", "envelope"],
        menu_icon="cast",
        default_index=0,
    )

# Home Page
if selected == "Home":
    st.title("Autism Spectrum Disorder Information")
    st.write("---")
    with st.container():
        col1, col2 = st.columns([3, 2])
        with col1:
            st.title("What is Autism Spectrum Disorder?")
            st.write("""
            Autism spectrum disorder (ASD) is a developmental disability caused by differences in the brain. People with ASD often have problems with social communication and interaction, and restricted or repetitive behaviors or interests. People with ASD may also have different ways of learning, moving, or paying attention.
            """)
        with col2:
            img1 = Image.open("image/asd_child.jpg")
            st.image(img1, width=300)

    with st.container():
        col1, col2 = st.columns([4, 2])
        with col1:
            st.title("What Causes Autism Spectrum Disorder?")
            st.write("""
            The Autism Spectrum Disorder Foundation lists the following as possible causes of ASD:

            :blue[Genetics] : Research suggests that ASD can be caused by a combination of genetic and environmental factors. Some genes have been identified as being associated with an increased risk for ASD, but no single gene has been proven to cause ASD.

            :blue[Environmental factors] : Studies are currently underway to explore whether certain exposure to toxins during pregnancy or after birth can increase the risk for developing ASD.

            :blue[Brain differences] : Differences in certain areas of the brain have been observed in people with ASD, compared to those without ASD. It is not yet known what causes these differences.
            """)
        with col2:
            img1 = Image.open("image/causes-of-autism.png")
            st.image(img1, width=350, caption="Causes of ASD")

    with st.container():
        col1, col2 = st.columns([4, 2])
        with col1:
            st.title("Symptoms of ASD:")

            st.write("""
            1. Avoids or does not keep eye contact
            2. Does not respond to name by 9 months of age
            3. Does not show facial expressions like happy, sad, angry, and surprised by 9 months of age
            4. Lines up toys or other objects and gets upset when order is changed
            5. Repeats words or phrases over and over (called echolalia)
            6. Plays with toys the same way every time
            7. Delayed language skills
            8. Delayed movement skills
            9. Delayed cognitive or learning skills
            10. Hyperactive, impulsive, and/or inattentive behavior
            11. Epilepsy or seizure disorder
            12. Unusual eating and sleeping habits
            13. Gastrointestinal issues (for example, constipation)
            14. Unusual mood or emotional reactions
            15. Anxiety, stress, or excessive worry
            16. Lack of fear or more fear than expected, etc.
            """)
            st.write("[Learn More >](https://www.who.int/news-room/fact-sheets/detail/autism-spectrum-disorders)")
        with col2:
            img = Image.open("image/autism.png")
            st.image(img, caption="Signs of ASD")
            img1 = Image.open("image/Strategies.jpeg")
            st.image(img1, caption="")

    # ---- WHAT I DO ----
    with st.container():
        left_column, right_column = st.columns([4, 2])
        with left_column:
            st.title("Relevent statistics ")

            st.write("""
                The exact prevalence of Autism Spectrum Disorder (ASD) in India is not well-established due to a lack of nationwide studies and consistent diagnostic criteria. However, some studies have estimated that the prevalence of ASD in India is between 1 and 2 per 1000 children.
                A recent study published in the Indian Journal of Pediatrics in 2020 estimated the prevalence of ASD in children aged 2 to 9 years in Kolkata, India, to be 1.25%. Another study published in the Journal of Autism and Developmental Disorders in 2018 found a prevalence of 0.64% among school-aged children in Chennai, India.

                • Prevalance of Autism: Between 1 in 500 (2/1,000) to 1 in 166 children (6/1,000) have an Autism Spectrum Disorder (Center for Disease Control).

                • Prevalance Rate: Approx. 1 in 500 or 0.20% or more than 2,160,000 people in India.

                • Incidence Rate: Approx. 1 in 90,666 or 11,914 people in India.

                • Incidence extrapolations for India for Autism: 11,914 per year, 250 per month, 57 per week, 8 per day, 1.4 per hour.

                • Autism is four times more prevalent in boys than girls in the US (Autism Society of America).

                • Autism is more common than Down Syndrome, which occurs in 1 out of 800 births.

                • The rate of incidence of autism is increasing 10-17% per year in the US (Autism Society of America).

                • Prevalence of autism is expected to reach 4 million people in the next decade in the US (Autism Society of America).
                """)

        with right_column:
            img = Image.open("image/autism-stats-1.jpg")
            st.image(img, width=350, caption="ASD ststistics")
            img1 = Image.open("image/autism-stats-2.png")
            st.image(img1, width=350, caption="USA data over 18 years")

    with st.container():
        st.title("World Autism Awareness Day")
        st.write(
            "This year, WAAD will be observed with a virtual event on Sunday, 2 April, from 10:00 a.m. to 1:00 p.m. EDT.The event is organized in close collaboration with autistic people and will feature autistic people from around the world discussing how the transformation in the narrative around neurodiversity can continue to be furthered in order to overcome barriers and improve the lives of autistic people. It will also address the contributions that autistic people make – and can make – to society, and to the achievement of the Sustainable Development Goals."
        )
    c1, c2 = st.columns([5, 5])
    im = Image.open("image/World.png")
    c1.image(im, caption="")
    im1 = Image.open("image/Worlds.png")

    c2.image(im1, caption="")

    # Add other sections of Home Page here

# Q&A Chatbot Page
elif selected == "Q&A Chatbot":
    st.title("Q&A Chatbot")
    st.write("---")


    def get_gemini_response(question):
        response = chat.send_message(question, stream=True)
        return response


    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    input = st.text_input("Input:", key="input")
    submit = st.button("Ask the question")

    if submit and input:
        response = get_gemini_response(input)
        st.session_state["chat_history"].append(("You", input))
        for chunk in response:
            st.write(chunk.text)
            st.session_state["chat_history"].append(("bot", chunk.text))

    st.subheader("Chat History")
    for role, text in st.session_state["chat_history"]:
        st.write(f"{role}: {text}")

# Statistics Page
elif selected == "Statistics":
    st.title("ASD Statistics")
    st.write("---")
    df = pd.read_csv("data_csv.csv")
    ASD_traits_data = df["ASD_traits"].unique().tolist()
    select_date = st.selectbox("ASD traits?", ASD_traits_data)
    df_up = df[df["ASD_traits"].isin(ASD_traits_data)]
    sub_opt = df_up["Sex"].unique().tolist()
    select_sub = st.multiselect("Gender", sub_opt)
    df_up_sub = df_up[df_up["Sex"].isin(select_sub)]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Jaundice statistics")
        with st.expander("See the plot"):
            fig = px.bar(df_up_sub, x="Sex", color="Jaundice")
            fig.update_layout(height=500, width=200)
            st.write(fig)

    with col2:
        st.subheader("Childhood Autism Rating Scale statistics")
        with st.expander("See the plot"):
            fig = px.bar(df_up_sub, x="Sex", color="Childhood Autism Rating Scale")
            fig.update_layout(height=500, width=200)
            st.write(fig)

    # Add other statistics plots here

# ASD Assessment Page
elif selected == "ASD Assessment":
    st.title("Autism Data Assessment")
    st.write("---")
    st.write("Fill the form below to check if your child is suffering from ASD ")
    autism_dataset = pd.read_csv('asd_data_csv.csv')

    X = autism_dataset.drop(columns='Outcome', axis=1)
    Y = autism_dataset['Outcome']
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    Y = autism_dataset['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)


    def ValueCount(str):
        return 1 if str == "Yes" else 0


    def Sex(str):
        return 1 if str == "Female" else 0


    # Form layout
    d1 = list(range(11))
    val1 = st.selectbox("Social Responsiveness", d1)
    d2 = list(range(19))
    val2 = st.selectbox("Age", d2)
    d3 = ["No", "Yes"]
    val3 = ValueCount(st.selectbox("Speech Delay", d3))
    val4 = ValueCount(st.selectbox("Learning disorder", d3))
    val5 = ValueCount(st.selectbox("Genetic disorders", d3))
    val6 = ValueCount(st.selectbox("Depression", d3))
    val7 = ValueCount(st.selectbox("Intellectual disability", d3))
    val8 = ValueCount(st.selectbox("Social/Behavioural issues", d3))
    val9 = ValueCount(st.selectbox("Anxiety disorder", d3))
    d4 = ["Female", "Male"]
    val10 = Sex(st.selectbox("Gender", d4))
    val11 = ValueCount(st.selectbox("Suffers from Jaundice", d3))
    val12 = ValueCount(st.selectbox("Family member history with ASD", d3))

    input_data = [val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(std_data)

    with st.expander("Analyze provided data"):
        st.subheader("Results:")
        if prediction[0] == 0:
            st.info('The person is not with Autism spectrum disorder')
        else:
            st.warning('The person is with Autism spectrum disorder')

# Contact Us Page
elif selected == "Contact Us":
    st.title("Get In Touch With Us!")
    contact_form = """
    <form action="https://formsubmit.co/YOUREMAIL@EMAIL.COM" method="POST">
         <input type="hidden" name="_captcha" value="false">
         <input type="text" name="name" placeholder="Your name" required>
         <input type="email" name="email" placeholder="Your email" required>
         <textarea name="message" placeholder="Your message here"></textarea>
         <button type="submit">Send</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)


    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    # Provide the correct path to your CSS file
    local_css("styles.css")

# Login/Signup Page
elif selected == "Login/Signup":
    selected_auth = option_menu(
        menu_title=None,
        options=["Login", "Signup"],
        icons=["box-arrow-in-right", "person-plus"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )
    if selected_auth == "Login":
        st.subheader("Login Section")
        username = st.text_input("User Name")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username, check_hashes(password, hashed_pswd))
            if result:
                st.success("Logged In as {}".format(username))
            else:
                st.warning("Incorrect Username/Password")
    elif selected_auth == "Signup":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')
        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user, make_hashes(new_password))
            st.success("You have successfully created an account")
            st.info("Go to Login Menu to login")

# Ensure proper closing of SQLite connection
conn.close()
