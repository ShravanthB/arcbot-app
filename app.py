import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import io
import re
import os
from dotenv import load_dotenv

api_key = st.secrets.get("OPENAI_API_KEY")

st.set_page_config(page_title="ArcBot" , page_icon="🍪")

st.html("""
<style>
[data-testid="stDecoration"] {
    background-image: linear-gradient(90deg, #AAFF00, #AAFF00);
    height: 6px;
}
</style>
""")

st.header("ArcBot Artifact Locator")

MAX_CHARS = 3900

def split_into_chunks(text, max_chars=MAX_CHARS):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

prompt_base = """You are a virtual assistant named ArcBot who is helping archaeologists predict promising locations to dig for artifacts. You also can give assumptions 
of which civilisations artifacts a user can find in the location that they enter.Ideally use only the data given below, however if there is no data for a question,
 feel free to add in accurate data! Remember to take into account that you can just dig up any castle ruins, roads and buildings!

"""


with st.sidebar:
    st.title("LGS_tufCookies")

    voice_enabled = st.checkbox("Enable voice input and output", value=False)

    if voice_enabled:
        audio_input = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            key="mic"
        )

        if audio_input:
            st.session_state["audio_input"] = audio_input

    if st.button("Reset Conversation"):
        st.session_state.messages = [{"role": "system", "content": prompt_base}]
        st.session_state.audio_input = None
        st.rerun()

if api_key:
    client = OpenAI(api_key=api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": prompt_base}]

    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    voice_prompt = None

    if voice_enabled:
        audio_data = st.session_state.get("audio_input")

        if audio_data:
            audio_file = io.BytesIO(audio_data["bytes"])
            audio_file.seek(0)
            audio_file.name = "speech.wav"

            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

            voice_prompt = transcript.text
            st.session_state.audio_input = None

    text_prompt = st.chat_input("Feel free to ask me questions about ancient civilisations!")

    final_input = text_prompt if text_prompt else voice_prompt

    if final_input:
        st.session_state.messages.append(
            {"role": "user", "content": final_input}
        )

        with st.chat_message("user"):
            st.markdown(final_input)

        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=st.session_state.messages,
            )

            answer = response.choices[0].message.content
            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            if voice_enabled:
                chunks = split_into_chunks(answer)
                audio_bytes = b""

                for chunk in chunks:
                    speech = client.audio.speech.create(
                        model="tts-1",
                        voice="nova",
                        input=chunk
                    )

                    audio_bytes += speech.read()

                st.audio(audio_bytes, format="audio/mp3", autoplay=True)

else:
    st.error("API key not found. Check your .env file.")