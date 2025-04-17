import streamlit as st
from streamlit_feedback import streamlit_feedback
import time


@st.cache_resource
def get_retriever_generator():
    from retrieval import retriever_generator

    return retriever_generator


@st.cache_resource
def get_db():
    from firestore import DBTransaction

    return DBTransaction()


@st.cache_resource
def get_service_config():
    from service_config import service_config

    return service_config


service_config = get_service_config()

retriever_generator = get_retriever_generator()
db = get_db()

possible_faces = {"😞": 1, "🙁": 2, "😐": 3, "🙂": 4, "😀": 5}


def bot_ui():
    if "last_prompt" not in st.session_state:
        st.session_state["last_prompt"] = ""
    if "response_generate" not in st.session_state:
        st.session_state["response_generate"] = None

    model = st.sidebar.selectbox(
        "Choose a Specialist", list(service_config.MODELS.keys())
    )
    st.title(f"Legal Specialist {model}")
    prompt = st.text_input("Input your query and press enter!")
    response_text = ""
    if prompt and prompt != st.session_state["last_prompt"]:
        with st.spinner(" "):
            response, st.session_state["is_valid"], st.session_state["extra_info"] = (
                retriever_generator.get_query_response(prompt, model)
            )
        if not st.session_state["is_valid"]:
            response_text = response
        else:
            response_text = response[
                12:
            ]  # Adjust this slicing based on actual response format
        datapoints = {
            "response_shown": response_text,
            "extra_info": st.session_state["extra_info"],
            "is_valid": st.session_state["is_valid"],
        }
        datapoints["timestamp"] = str(int(time.time()))
        db.insert(datapoints, collection="query_response")
        del datapoints

        st.session_state["response_generate"] = response_text
        st.session_state["last_prompt"] = prompt
    elif st.session_state["response_generate"]:
        response_text = st.session_state["response_generate"]

    st.write(response_text if response_text else "Submit a query to get a response.")

    feedback = streamlit_feedback(
        feedback_type="faces",
        key=prompt,
    )
    if feedback:
        datapoints = {
            "response_shown": response_text,
            "extra_info": st.session_state["extra_info"],
            "is_valid": st.session_state["is_valid"],
        }
        score = feedback["score"]
        st.toast(f"Feedback submitted: {score}")
        datapoints["score"] = possible_faces[score]
        datapoints["timestamp"] = str(int(time.time()))
        db.insert(datapoints, collection="users_feedback")
        del datapoints
        st.session_state["response_generate"] = None  # Reset after feedback

    # else:
    # print("fdkjfs dfsdklfjds l")
    # Add the disclaimer using custom HTML and CSS
    st.markdown(
        """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117; /* Dark background color similar to Streamlit's default dark mode */
        color: #ffffff; /* White text color for better readability in dark mode */
        text-align: center;
        padding: 10px;
        box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
        font-family: 'Source Sans Pro', sans-serif;
        z-index: 1000;
    }
    .footer p {
        margin: 0;
        font-size: 12px; /* Adjust font size to ensure it fits well */
        line-height: 1.5;
        max-width: 60%; /* Ensure the text is contained within a reasonable width */
        margin: auto; /* Center the text horizontally */
    }
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(81, 203, 238, 0.7);
        }

        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(81, 203, 238, 0);
        }

        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(81, 203, 238, 0);
        }
    }
    div.stSpinner>div>div {
        content: "";
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: rgba(81, 203, 238, 1);
        animation: pulse 2s infinite;
    }
    </style>
    <div class="footer">
        <p><strong>Disclaimer:</strong> It is advisable to consult with a tax professional to ensure compliance and optimal tax treatment. The above information is based on legal precendence available with us as on a particular date. Please refer to the latest provisions of the law as there may have been amendments relavent to your query after the date of decision.</div>
    """,
        unsafe_allow_html=True,
    )
bot_ui()
