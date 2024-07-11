import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI
import os
import pandas as pd
import fitz  # PyMuPDF
import re
from kbcstorage.client import Client
import uuid
from fuzzywuzzy import fuzz

# Load the OpenAI API key and Keboola credentials from secrets
token = st.secrets["storage_token"]
kbc_url = st.secrets["url"]
client = OpenAI(api_key=st.secrets["api_key"])

# Initialize Client
client_kbl = Client(kbc_url, token)

# Predefined user login credentials
users = {"admin": st.secrets["admin"]}


# Function to check login credentials
def check_credentials(password):
    return password == users["admin"]


# Function to change button colors
def change_button_color(font_color, background_color, border_color):
    button_html = f"""
    <style>
        .stButton > button {{
            color: {font_color};
            background-color: {background_color};
            border: 1px solid {border_color};
        }}
    </style>
    """
    st.markdown(button_html, unsafe_allow_html=True)


# Function to display logo ! when in Keboola, there is "./app/static/keboola.png", in local "./static/keboola.png"
LOGO_IMAGE_PATH = os.path.abspath("./app/static/keboola.png")

# Set page title and icon
st.set_page_config(
    page_title="Knowledge Base Data App",
    page_icon=LOGO_IMAGE_PATH,
)


# Function to hide custom anchor links
def hide_custom_anchor_link():
    st.markdown(
        """
        <style>
        /* Hide anchors directly inside custom HTML headers */
        h1 > a, h2 > a, h3 > a, h4 > a, h5 > a, h6 > a {
            display: none !important;
        }
        /* If the above doesn't work, it may be necessary to target by attribute if Streamlit adds them dynamically */
        [data-testid="stMarkdown"] h1 a, [data-testid="stMarkdown"] h2 a, [data-testid="stMarkdown"] h3 a, [data-testid="stMarkdown"] h4 a, [data-testid="stMarkdown"] h5 a, [data-testid="stMarkdown"] h6 a {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Add a function to normalize file names
def normalize_file_name(file_name):
    # Convert to lowercase and replace spaces with underscores
    return file_name.lower().replace(" ", "_")


# Fetching data from Keboola Storage --> dataframe
def get_dataframe(table_name):
    try:
        table_detail = client_kbl.tables.detail(table_name)
        client_kbl.tables.export_to_file(table_id=table_name, path_name="")
        with open("./" + table_detail["name"], mode="rt", encoding="utf-8") as in_file:
            lazy_lines = (line.replace("\0", "") for line in in_file)  # opraveno
            with open("data.csv", "w", encoding="utf-8") as out_file:
                for line in lazy_lines:
                    out_file.write(line)
        df = pd.read_csv("data.csv")
        return df
    except Exception as e:
        st.error(f"Error fetching dataframe: {e}")
        return pd.DataFrame()


# Function to write data to Keboola - incremental loading
def write_to_keboola(data, table_name, table_path, incremental):
    data.to_csv(table_path, index=False, compression="gzip")
    client_kbl.tables.load(
        table_id=table_name, file_path=table_path, is_incremental=incremental
    )


# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "home"

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "responses" not in st.session_state:
    st.session_state.responses = []

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if "file_page_map" not in st.session_state:
    st.session_state.file_page_map = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

if "annotation" not in st.session_state:
    st.session_state.annotation = ""


# Load extracted text from table "extrakt" in Keboola
def load_extracted_text_from_keboola():
    try:
        extracted_data = get_dataframe("in.c-RHLF-app.extrakt")
        if extracted_data.empty:
            return "", []
        # Replace NaN/None values with empty strings
        extracted_data["Extracted_Text"] = extracted_data["Extracted_Text"].fillna("")
        extracted_text = " ".join(extracted_data["Extracted_Text"].tolist())
        file_page_map = list(
            zip(
                extracted_data["File_Name"],
                extracted_data["Page_Number"],
                extracted_data["Extracted_Text"],
            )
        )
        return extracted_text, file_page_map
    except Exception as e:
        st.error(f"Error loading extracted text: {e}")
        return "", []


# Function to extract text from PDF files and keep track of the file and page number
def extract_text_from_pdfs(files):
    extracted_text = ""
    file_page_map = []
    english_text_pattern = re.compile(r'[a-zA-Z0-9\s.,;:!?\'"-]+')

    for file in files:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text("text")
            english_text = " ".join(english_text_pattern.findall(text))
            extracted_text += english_text
            file_page_map.append(
                (normalize_file_name(file.name), page_num + 1, english_text)
            )
    return extracted_text, file_page_map


# This function searches for a previously answered question in the feedback database by calculating the similarity between the current question and stored questions (with fuzzy wuzzy).
# It returns the best matching response and its source if a high similarity is found.
def search_feedback(question, feedback_scores):
    try:
        feedback_data = get_dataframe("in.c-RHLF-app.feedback")
        responses = []
        for index, row in feedback_data.iterrows():
            similarity = (
                fuzz.ratio(row["Question"].strip().lower(), question.strip().lower())
                / 100
            )
            if similarity >= 0.9:
                response = row["Response"]
                source = row["File_Name"]
                score = feedback_scores.get(response, {"total": 0, "positive": 0})
                responses.append(
                    (response, source, score["total"], score["positive"], similarity)
                )
        if responses:
            # Calculate scores
            feedback_scores = []
            for response, source, total, positive, similarity in responses:
                score = positive / total if total > 0 else 0
                feedback_scores.append((response, source, total, score, similarity))

            # Sort responses by total feedback, score, and similarity
            feedback_scores.sort(
                key=lambda x: (
                    -x[2],
                    -x[3],
                    -x[4],
                )  # Prioritize by total feedback, then positive feedback percentage, then similarity
            )

            # Select responses based on the new criteria
            best_responses = []
            for response, source, total, score, similarity in feedback_scores:
                if total > 0:  # Ensure there is at least one feedback
                    best_responses.append((response, source, total, score, similarity))
                if len(best_responses) >= 3:  # Limit to top 3 responses
                    break

            if best_responses:
                # Return the best response based on the new criteria
                best_response = best_responses[0][0]
                best_source = best_responses[0][1]
                return best_response, best_source

    except Exception as e:
        return None, None
    return None, None


# Function to upload file to Keboola Storage
def upload_file_to_keboola(file):
    normalized_file_name = normalize_file_name(file.name)
    with open(normalized_file_name, "wb") as f:
        f.write(file.getbuffer())
    file_path = os.path.abspath(normalized_file_name)
    client_kbl.files.upload_file(file_path)
    os.remove(file_path)  # Clean up the local file after upload


# Function to get list of uploaded files from Keboola Storage
def list_uploaded_files():
    files = client_kbl.files.list()
    return [
        (file["id"], file["name"], file["sizeBytes"])
        for file in files
        if file["name"].endswith(".pdf")
    ]


# Function to delete file from Keboola Storage
def delete_file_from_keboola(file_id, file_name=None):
    client_kbl.files.delete(file_id)
    if file_name:
        delete_extracted_text(file_name)
        delete_feedback_entries(file_name)


# Update the delete_extracted_text function to use the normalized file names
def delete_extracted_text(file_name):
    try:
        # Normalize the file name
        normalized_file_name = normalize_file_name(file_name)

        # Fetch the data
        extracted_data = get_dataframe("in.c-RHLF-app.extrakt")

        # Ensure 'File_Name' column exists
        if "File_Name" not in extracted_data.columns:
            st.error("Column 'File_Name' not found in extracted data.")
            return

        # Filter out the rows where File_Name matches the normalized file_name to delete
        updated_data = extracted_data[
            extracted_data["File_Name"].apply(normalize_file_name)
            != normalized_file_name
        ]

        # Write the updated data back to Keboola
        write_to_keboola(
            updated_data, "in.c-RHLF-app.extrakt", f"extrakt.csv.gz", False
        )

        # Reload the session state with the updated data
        st.session_state.extracted_text, st.session_state.file_page_map = (
            load_extracted_text_from_keboola()
        )
        st.success(f"Extracted text for {file_name} deleted successfully.")
    except Exception as e:
        st.error(f"Error deleting extracted text: {e}")


# Update the delete_feedback_entries function to use the normalized file names
def delete_feedback_entries(file_name):
    try:
        # Normalize the file name
        normalized_file_name = normalize_file_name(file_name)

        feedback_data = get_dataframe("in.c-RHLF-app.feedback")
        updated_feedback = feedback_data[
            feedback_data["File_Name"].apply(normalize_file_name)
            != normalized_file_name
        ]
        write_to_keboola(
            updated_feedback, "in.c-RHLF-app.feedback", f"feedback.csv.gz", False
        )
    except Exception as e:
        st.error(f"Error deleting feedback entries: {e}")


# Function to delete all files from Keboola Storage and clear extracted text
# It deletes: feedbacks, extracted text, file
def delete_all_files_and_text():
    files = client_kbl.files.list()
    for file in files:
        if file["name"].endswith(".pdf"):
            client_kbl.files.delete(file["id"])
    extracted_data = pd.DataFrame(
        columns=["File_Name", "Page_Number", "Extracted_Text"]
    )
    write_to_keboola(extracted_data, "in.c-RHLF-app.extrakt", f"extrakt.csv.gz", False)
    feedback_data = pd.DataFrame(
        columns=["Feedback", "Question", "Response", "File_Name", "Comment"]
    )
    write_to_keboola(feedback_data, "in.c-RHLF-app.feedback", f"feedback.csv.gz", False)
    st.session_state.extracted_text = ""
    st.session_state.file_page_map = []
    st.session_state.feedback = {}
    st.session_state.annotation = ""


# Function to update feedback scores by counting the total number of feedbacks and the number of positive feedback of each response
# Positive feedback is when: "Great" or "Correct information, but not what i was looking for." -> in the second options there is then some improvements in generated response
def update_feedback_scores(feedback_data):
    feedback_scores = {}
    for index, row in feedback_data.iterrows():
        response = row["Response"]
        feedback = row["Feedback"]
        if response not in feedback_scores:
            feedback_scores[response] = {"total": 0, "positive": 0}
        feedback_scores[response]["total"] += 1
        if feedback in [
            "Great!",
            "Correct information, but not what I was looking for.",
        ]:
            feedback_scores[response]["positive"] += 1
    return feedback_scores


# Function to display the home page - text input for asking questions, responses to the questions along with options for users to give feedback on the answers
# Homepage includes annotation.
def show_home_page():
    # Ensure extracted text is loaded
    if "extracted_text" not in st.session_state or not st.session_state.extracted_text:
        st.session_state.extracted_text, st.session_state.file_page_map = (
            load_extracted_text_from_keboola()
        )

    # Ensure annotation is generated if not already present
    if "annotation" not in st.session_state or not st.session_state.annotation:
        st.session_state.annotation = generate_annotation(
            st.session_state.extracted_text
        )

    # Load feedback scores
    feedback_data = get_dataframe("in.c-RHLF-app.feedback")
    feedback_scores = update_feedback_scores(feedback_data)

    # Chat application
    question = st.text_input(
        "Ask your questions to the Knowledge Base Data App:", key="question_input"
    )

    ask_button = st.button("Ask question", key="ask_button")
    change_button_color(
        "#FFFFFF", "#1EC71E", "#1EC71E"
    )  # Green background and white font color

    # Display the annotation
    st.info(f"**What is in this knowledge base:** {st.session_state.annotation}")

    if ask_button:
        if question:
            # Check if the question was already answered
            previous_answer, previous_source = search_feedback(
                question, feedback_scores
            )
            unique_id = str(uuid.uuid4())
            if previous_answer:
                st.session_state.responses.insert(
                    0,
                    (question, previous_answer, "", previous_source, False, unique_id),
                )
            else:
                if st.session_state.extracted_text:
                    new_answer, source = generate_response(question)
                    feedback_comment = ""  # Initialize feedback_comment variable
                    if (
                        new_answer
                        == "Unfortunately, this answer is not in my knowledge database."
                    ):
                        st.session_state.responses.insert(
                            0, (question, new_answer, "", source, True, unique_id)
                        )
                    else:
                        st.session_state.responses.insert(
                            0, (question, new_answer, "", source, False, unique_id)
                        )
                        write_feedback_to_file(
                            "", question, new_answer, source, feedback_comment
                        )  # Save the response to feedback with empty comment
                else:
                    st.warning(
                        "Please upload some files first and extract text from them."
                    )
            st.rerun()

    if st.session_state.responses:
        st.subheader("Responses:")
        for i, (q, r, linked_text, source, feedback_given, unique_id) in enumerate(
            st.session_state.responses
        ):
            with st.container(border=True):
                st.write(f"**Question:** {q}")
                st.write(f"**Answer:** {r}")
                if source:
                    st.write(
                        f"**Source:** {source}"
                    )  # Display source only if it exists
                if linked_text:
                    with st.expander("View document excerpt"):
                        st.markdown(
                            f"<div style='background-color: yellow;'>{linked_text}</div>",
                            unsafe_allow_html=True,
                        )

                feedback_comment = ""  # Initialize feedback_comment variable

                if (
                    not feedback_given
                    and r
                    != "Unfortunately, this answer is not in my knowledge database."
                ):
                    feedback_options = [
                        ("🌟", "Great!"),
                        ("👍", "Correct information, but not what I was looking for."),
                        ("😐", "Neutral."),
                        ("👎", "Misleading or incorrect answer."),
                        ("🚫", "Completely off."),
                    ]

                    # Apply custom CSS to buttons
                    st.markdown(
                        """
                        <style>
                        .stButton>button {
                            background-color: white;
                            color: black;
                            border: 1px solid #d3d3d3; /* Light grey color */
                            padding: 5px 10px;
                            border-radius: 5px;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    for emoji, label in feedback_options:
                        feedback_col, feedback_text_col = st.columns([1, 5])
                        with feedback_col:
                            if st.button(emoji, key=f"{emoji}_button_{unique_id}"):
                                feedback_comment = st.session_state.get(
                                    f"comment_{unique_id}", ""
                                )
                                st.session_state.responses[i] = (
                                    q,
                                    r,
                                    linked_text,
                                    source,
                                    True,
                                    unique_id,
                                )
                                st.session_state.feedback[unique_id] = label
                                write_feedback_to_file(
                                    label, q, r, source, feedback_comment
                                )
                                st.rerun()
                        with feedback_text_col:
                            st.markdown(
                                f"""
                                <div style="display: flex; align-items: center; height: 100%; margin-left: 5px;"> <!-- Reduced margin-left to 5px -->
                                    <span style="margin: auto 0;">{label}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    feedback_comment = st.text_area(
                        "Additional comment:", key=f"comment_{unique_id}"
                    )

                if unique_id in st.session_state.feedback:
                    st.write(f"**Feedback:** {st.session_state.feedback[unique_id]}")
                if feedback_comment:
                    st.write(f"**Comment:** {feedback_comment}")

    display_footer_section()


# Function extracts relevant text from the extracted data based on the similarity between the text and the question, returning the first 15,000 characters of the relevant text.
# Limit =  15000 characters
def extract_relevant_text(question):
    extracted_data = get_dataframe("in.c-RHLF-app.extrakt")
    relevant_text = ""

    # Ensure the question is a string
    if not isinstance(question, str):
        raise ValueError("The question must be a string")

    for index, row in extracted_data.iterrows():
        extracted_text = row["Extracted_Text"]

        # Ensure the extracted_text is a string
        if (
            isinstance(extracted_text, str)
            and fuzz.partial_ratio(extracted_text, question) > 50
        ):
            relevant_text += extracted_text + " "
    return relevant_text[:15000]  # Limit to first 15000 characters


# Function generates a response to the given question using OpenAI's GPT-4 model, utilizing the extracted relevant text and feedback comments. It includes context for the AI and finds the source of the response from the extracted data.
# It works with: extracted relevant text, feedbacks, and comments.
def generate_response(question, direction=None):
    # Extract relevant text from uploaded documents
    context = extract_relevant_text(question)
    feedback_comments = extract_feedback_comments(question)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an analyst. Use only the context provided in the uploaded documents to answer the questions. "
                "Do not use your opinions or external information. Cite as accurately as possible from the provided text. "
                "If the answer is not found in the provided documents, respond with 'Unfortunately, this answer is not in my knowledge database.' "
                "When citing the text, include the surrounding context to make it easy to find the source in the documents."
            ),
        },
        {"role": "user", "content": question},
        {"role": "system", "content": context},
    ]

    # Adding the context and feedback comments separately to avoid length issues
    if feedback_comments:
        messages.append({"role": "system", "content": feedback_comments})

    if direction == "follow":
        messages.insert(
            1,
            {
                "role": "system",
                "content": "Follow the previous direction but improve it significantly.",
            },
        )
    elif direction == "completely_new":
        messages.insert(
            1,
            {
                "role": "system",
                "content": "Ignore the previous direction and provide a completely new answer.",
            },
        )

    try:
        response = client.chat.completions.create(model="gpt-4", messages=messages)
        answer = response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Unfortunately, this answer is not in my knowledge database.", ""

    # Find the source document based on the similarity of the extracted text
    if "Unfortunately, this answer is not in my knowledge database." in answer:
        source = ""
    else:
        extracted_data = get_dataframe("in.c-RHLF-app.extrakt")
        source = "Source not found"
        highest_similarity = 0
        for index, row in extracted_data.iterrows():
            extracted_text = str(
                row["Extracted_Text"]
            )  # Ensure extracted_text is a string
            answer_text = str(answer)  # Ensure answer is a string
            similarity = fuzz.partial_ratio(extracted_text, answer_text)
            if similarity > highest_similarity:
                highest_similarity = similarity
                source = row["File_Name"]

    return answer, source


# Function extracts feedback comments from the feedback data that are similar to the given question
def extract_feedback_comments(question):
    feedback_data = get_dataframe("in.c-RHLF-app.feedback")
    comments = ""
    for index, row in feedback_data.iterrows():
        if fuzz.partial_ratio(row["Question"], question) > 50:
            comments += f"Comment: {row['Comment']} "
    return comments


# Function to generate an annotation using OpenAI
# Limit 3000 characters for summary
def generate_annotation(extracted_text):
    prompt = f"""
    Please provide a concise summary of the following text in one sentence.
    Dont use url and every time start with words: In this knowledge base are information about {extracted_text[:3000]}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating annotation: {e}")
        return ""


# Function to display the login page
def show_login_page():
    password = st.text_input("Password", type="password", key="login_password")
    login_button = st.button("Login", key="login_button")
    change_button_color(
        "#FFFFFF", "#1EC71E", "#1EC71E"
    )  # Green background and white font color
    if login_button:
        if check_credentials(password):
            st.session_state.logged_in = True
            st.session_state.page = "admin"  # Set the page to admin after login
            st.rerun()
        else:
            st.error("Invalid username or password")


# Function to reset session state and navigate to the home page
def reset_session_state():
    st.session_state.logged_in = False
    st.session_state.page = "home"
    st.session_state.uploaded_files = []
    st.session_state.responses = []
    st.session_state.extracted_text = ""
    st.session_state.file_page_map = []
    st.session_state.feedback = {}
    st.session_state.annotation = ""
    st.rerun()  # Navigate to 'home' page


# Function to display the admin page displays the admin page where users can upload PDF files, view uploaded files, delete files, and see feedback statistics.
def show_admin_page():
    # Add file uploader
    uploaded_files = st.file_uploader(
        "Upload your PDF files (without diacritics in file name)", accept_multiple_files=True, key="file_uploader"
    )
    if uploaded_files:
        if st.button("Upload PDF", key="upload_button"):
            for file in uploaded_files:
                st.session_state.uploaded_files.append(file)
                upload_file_to_keboola(file)

            # Extract text from the uploaded files
            extracted_text, file_page_map = extract_text_from_pdfs(uploaded_files)
            st.session_state.extracted_text += (
                extracted_text  # Append new text to existing text
            )
            st.session_state.file_page_map.extend(
                file_page_map
            )  # Append new file_page_map to existing map
            st.success("Files uploaded and text extracted successfully!")
            # Save extracted text to Keboola
            save_extracted_text_to_keboola(st.session_state.file_page_map)

    # Display list of uploaded files in a container
    with st.container(border=True):
        st.subheader("All uploaded files:")

        try:
            uploaded_files_list = list_uploaded_files()
            st.session_state.uploaded_files = [
                (file_id, normalize_file_name(file_name), file_size)
                for file_id, file_name, file_size in uploaded_files_list
            ]  # Update the session state with the latest file list
            for file_id, file_name, file_size in st.session_state.uploaded_files:
                file_col, delete_col = st.columns([4, 1])
                with file_col:
                    st.write(f"{file_name} (Size: {file_size} bytes)")
                with delete_col:
                    if st.button("Delete", key=f"delete_button_{file_id}"):
                        delete_file_from_keboola(file_id, file_name)
                        st.rerun()
        except Exception as e:
            st.error(f"Error loading uploaded files: {e}")

    if st.button("Delete all files & delete all text", key="delete_all_button"):
        delete_all_files_and_text()
        st.success("All files and extracted text have been deleted.")
        st.rerun()

    st.header("Statistics")
    st.subheader("Feedback Summary")

    feedback_data = get_dataframe("in.c-RHLF-app.feedback")
    feedback_data["Count"] = 1

    feedback_options = {
        "Great!": "🌟",
        "Correct information, but not what I was looking for.": "👍",
        "Neutral.": "😐",
        "Misleading or incorrect answer.": "👎",
        "Completely off.": "🚫",
    }

    feedback_data["Feedback_Emoji"] = feedback_data["Feedback"].map(feedback_options)

    feedback_summary = (
        feedback_data.groupby(["File_Name", "Feedback_Emoji"])
        .agg(Answer_Count=pd.NamedAgg(column="Count", aggfunc="sum"))
        .unstack(fill_value=0)
    )

    feedback_summary.columns = feedback_summary.columns.droplevel(
        0
    )  # Drop the multi-index

    st.dataframe(feedback_summary)

    # Display feedback comments
    st.subheader("Feedback Comments")
    feedback_comments = feedback_data[feedback_data["Comment"].notnull()]
    feedback_comments = feedback_comments[
        ["Comment", "Feedback", "Question", "Response", "File_Name"]
    ]
    st.dataframe(feedback_comments)

    display_footer_section()


# Function to write feedback to a file - writes new feedback data to a temporary CSV file and then uploads it to the feedback table in Keboola Storage using incremental loading.
def write_feedback_to_file(feedback, question, response, file_name, comment=""):
    if comment is None:
        comment = ""  # Ensure comment is a string if it's None

    # Create a DataFrame for the new feedback entry
    new_feedback_data = pd.DataFrame(
        [[feedback, question, response, file_name, comment]],
        columns=["Feedback", "Question", "Response", "File_Name", "Comment"],
    )

    # Write the new feedback entry to a temporary CSV file
    new_feedback_data.to_csv("feedback.csv", index=False)

    # Append the new feedback entry to the existing Keboola feedback table
    client_kbl.tables.load(
        table_id="in.c-RHLF-app.feedback", file_path="feedback.csv", is_incremental=True
    )


def save_extracted_text_to_keboola(file_page_map):
    try:
        # Fetch existing data
        existing_data = get_dataframe("in.c-RHLF-app.extrakt")

        # Create a DataFrame for the new data
        new_data = pd.DataFrame(
            {
                "File_Name": [normalize_file_name(f[0]) for f in file_page_map],
                "Page_Number": [f[1] for f in file_page_map],
                "Extracted_Text": [f[2] for f in file_page_map],
            }
        )

        # Append new data to existing data
        combined_data = pd.concat([existing_data, new_data])

        # Remove duplicates based on 'File_Name' and 'Page_Number'
        combined_data.drop_duplicates(
            subset=["File_Name", "Page_Number"], keep="last", inplace=True
        )

        # Save the combined data back to Keboola
        write_to_keboola(
            combined_data, "in.c-RHLF-app.extrakt", f"extrakt.csv.gz", False
        )

        st.success("Extracted text saved to Keboola successfully.")
    except Exception as e:
        st.error(f"Error saving extracted text to Keboola: {e}")

        st.success("Extracted text saved to Keboola successfully.")
    except Exception as e:
        st.error(f"Error saving extracted text to Keboola: {e}")


# Display logo and title
st.image(LOGO_IMAGE_PATH)
hide_img_fs = """
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        """
st.markdown(hide_img_fs, unsafe_allow_html=True)

st.title("RLHF Knowledge Base Data App")

# Menu at the top of the page using streamlit-option-menu
selected = option_menu(
    menu_title=None,
    options=["App", "Admin"],
    icons=["house", "box-arrow-in-right"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)


# Function to display footer
def display_footer_section():
    st.markdown(
        """
    <style>
    .footer {
        width: 100%;
        font-size: 14px; /* Adjust font size as needed */
        color: #22252999; /* Adjust text color as needed */
        padding: 10px 0;  /* Adjust padding as needed */
        display: flex; 
        justify-content: space-between;
        align-items: center;
    }
    .footer p {
        margin: 0;  /* Removes default margin for p elements */
        padding: 0;  /* Ensures no additional padding is applied */
    }
    </style>
    <div class="footer">
        <p>(c) Keboola 2024</p>
        <p>Version 2.0</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# Set the page state based on the selected menu option
if selected == "App":
    st.session_state.page = "home"
elif selected == "Admin" and not st.session_state.logged_in:
    st.session_state.page = "login"
elif selected == "Admin" and st.session_state.logged_in:
    st.session_state.page = "admin"

# Determine which page to display
if st.session_state.page == "login" and not st.session_state.logged_in:
    show_login_page()
elif st.session_state.page == "admin" and st.session_state.logged_in:
    show_admin_page()  # Show the admin page
else:
    show_home_page()

# Hide custom anchor links
hide_custom_anchor_link()
