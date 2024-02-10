from typing import Dict, List, Union

import numpy as np
import openai
import pandas as pd
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from scipy.spatial.distance import cosine

openai.api_key = st.secrets["OPENAI_API_KEY"]


def merge_dataframes(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Merges a list of DataFrames, keeping only specific columns."""
    # Concatenate the list of dataframes
    combined_dataframe = pd.concat(
        dataframes, ignore_index=True
    )  # Combine all dataframes into one

    # Ensure that the resulting dataframe only contains the columns "context", "questions", "answers"
    combined_dataframe = combined_dataframe[
        ["context", "questions", "answers"]
    ]  # Filter for specific columns

    return combined_dataframe  # Return the merged and filtered DataFrame


def call_chatgpt(prompt: str) -> str:
    """
    Uses the OpenAI API to generate an AI response to a prompt.

    Args:
        prompt: A string representing the prompt to send to the OpenAI API.

    Returns:
        A string representing the AI's generated response.

    """

    # Use the OpenAI API to generate a response based on the input prompt.
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Extract the text from the first (and only) choice in the response output.
    ans = response.choices[0]["text"]

    # Return the generated AI response.
    return ans


def openai_text_embedding(prompt: str) -> str:
    return openai.Embedding.create(input=prompt, model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]


def calculate_sts_openai_score(sentence1: str, sentence2: str) -> float:
    # Compute sentence embeddings
    embedding1 = openai_text_embedding(sentence1)  # Flatten the embedding array
    embedding2 = openai_text_embedding(sentence2)  # Flatten the embedding array

    # Convert to array
    embedding1 = np.asarray(embedding1)
    embedding2 = np.asarray(embedding2)

    # Calculate cosine similarity between the embeddings
    similarity_score = 1 - cosine(embedding1, embedding2)

    return similarity_score


def add_dist_score_column(
    dataframe: pd.DataFrame,
    sentence: str,
) -> pd.DataFrame:
    dataframe["stsopenai"] = dataframe["questions"].apply(
        lambda x: calculate_sts_openai_score(str(x), sentence)
    )

    sorted_dataframe = dataframe.sort_values(by="stsopenai", ascending=False)
    return sorted_dataframe.iloc[:5, :]


def convert_to_list_of_dict(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Reads in a pandas DataFrame and produces a list of dictionaries with two keys each, 'question' and 'answer.'

    Args:
        df: A pandas DataFrame with columns named 'questions' and 'answers'.

    Returns:
        A list of dictionaries, with each dictionary containing a 'question' and 'answer' key-value pair.
    """

    # Initialize an empty list to store the dictionaries
    result = []

    # Loop through each row of the DataFrame
    for index, row in df.iterrows():
        # Create a dictionary with the current question and answer
        qa_dict_quest = {"role": "user", "content": row["questions"]}
        qa_dict_ans = {"role": "assistant", "content": row["answers"]}

        # Add the dictionary to the result list
        result.append(qa_dict_quest)
        result.append(qa_dict_ans)

    # Return the list of dictionaries
    return result


# file_names = [f"output_files/file_{i}.txt" for i in range(131)]
file_names = [f"output_files_large/file_{i}.txt" for i in range(1310)]


# Initialize an empty list to hold all documents
all_documents = []  # this is just a copy, you don't have to use this

# Iterate over each file and load its contents
for file_name in file_names:
    loader = TextLoader(file_name)
    documents = loader.load()
    all_documents.extend(documents)

# Split the loaded documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(all_documents)

# Create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# embedding_function = SentenceTransformer("all-MiniLM-L6-v2")
# embedding_function = openai_text_embedding

# Load the documents into Chroma
db = Chroma.from_documents(docs, embedding_function)


st.title("Youth Homelessness Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.sidebar.markdown("""This is an app to help you navigate the website of YSA""")

clear_button = st.sidebar.button("Clear Conversation", key="clear")

if clear_button:
    st.session_state.messages = []

# React to user input
if prompt := st.chat_input("Tell me about YSA"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    question = prompt

    docs = db.similarity_search(question)
    docs_2 = db.similarity_search_with_score(question)
    docs_2_table = pd.DataFrame(
        {
            "source": [docs_2[i][0].metadata["source"] for i in range(len(docs))],
            "content": [docs_2[i][0].page_content for i in range(len(docs))],
            "distances": [docs_2[i][1] for i in range(len(docs))],
        }
    )
    ref_from_db_search = docs_2_table["content"]

    engineered_prompt = f"""
        Based on the context: {ref_from_db_search},
        answer the user question: {question}.
        Answer the question directly (don't say "based on the context, ...")
    """

    answer = call_chatgpt(engineered_prompt)
    response = answer

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Wait for it..."):
            st.markdown(response)
            with st.expander("See reference:"):
                st.table(docs_2_table)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.messages.append(
        {"role": "assistant", "content": docs_2_table.to_json()}
    )
