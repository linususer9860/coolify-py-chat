import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Loading environment variables from .env file
load_dotenv()

# Function to initialize conversation chain with GROQ language model
groq_api_key = os.environ['GROQ_API_KEY']

# Initializing GROQ chat with provided API key, model name, and settings
llm_groq = ChatGroq(
    groq_api_key=groq_api_key, model_name="llama3-70b-8192",
    temperature=0.2
)

@cl.on_chat_start
async def start():
    files = None  # Initialize variable to store uploaded files

    # Wait for the user to upload files
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload one or more pdf files to begin!",
            accept=["application/pdf"],
            max_size_mb=100,  # Optionally limit the file size
            max_files=10,
            timeout=180,  # Set a timeout for user response
        ).send()

    # Process each uploaded file
    texts = []
    metadatas = []
    for file in files:
        logging.info(f"Processing file: {file.name}")

        # Read the PDF file
        try:
            pdf = PyPDF2.PdfReader(file.content)
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text() or ""
        except Exception as e:
            await cl.Message(content=f"Error reading {file.name}: {str(e)}").send()
            continue

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)

        # Create metadata for each chunk
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Inform the user that processing has ended
    msg = cl.Message(content=f"Processing {len(files)} files done. You can now ask questions!")
    await msg.send()

    # Store the chain in user session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    # Call the chain with user's message content
    try:
        res = await chain.ainvoke(message.content, callbacks=[cb])
        answer = res["answer"]
        source_documents = res["source_documents"]

        text_elements = []  # Initialize list to store text elements

        # Process source documents if available
        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]

            # Add source references to the answer
            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"

        # Return results
        await cl.Message(content=answer, elements=text_elements).send()
    except Exception as e:
        await cl.Message(content=f"Error processing your request: {str(e)}").send()

# The script will be run by Chainlit automatically