import os
import streamlit as st

from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.chains import LLMChain
from langchain.llms import Anyscale
from langchain.prompts import PromptTemplate


def extract_audio(url):
    audio_dir = "./audio/"

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    # Extract audio from video on Youtube, store in local file
    yt_audio_loader = YoutubeAudioLoader([url], audio_dir)
    blobs = yt_audio_loader.yield_blobs()
    blobs = list(blobs) # Making the generator object into a list triggers the audio file download
    
    return blobs[-1] # Blobs seems to persist between function calls, so this list grows, meaning we have to return the last added blob

def transcribe_audio(path):
    transcriber = AssemblyAIAudioTranscriptLoader(file_path=path, api_key="0b74cb52b0c84aeca5db0492a3170b0c")
    docs = transcriber.load()
    return docs[0].page_content

def summarize_transcript(transcript):
    template = """You will be provided a transcription of a video. Your task is to generate a summary of the video. 
    Transcript: {transcript}"""

    prompt = PromptTemplate(template=template, input_variables=["transcript"])

    llm = Anyscale(
        model_name="meta-llama/Llama-2-70b-chat-hf",
        anyscale_api_key="esecret_hkb7lc3jszgdva8s3bm3qfgs9v",
        anyscale_api_base="https://api.endpoints.anyscale.com/v1",
        max_tokens=-1
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    summary = llm_chain.run(transcript)
    return summary

def summarize_yt_video(url):
    blob = extract_audio(url)
    transcript = transcribe_audio(str(blob.path))
    summary = summarize_transcript(transcript)
    return summary

st.title('YouTube Video Summarizer')
st.text("By Kevin Geertjens")
st.subheader("Generate a summary of any YouTube video you want.")

url = st.text_input("Video URL")
clicked = st.button("Summarize")

st.header("Summary")
summary = ""
if clicked:
    with st.spinner("Summarizing..."):
        summary = summarize_yt_video(url)

st.markdown(summary)