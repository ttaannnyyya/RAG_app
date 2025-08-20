from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableSequence, RunnableLambda
#for sengmailtool
from langchain_community.tools.gmail.send_message import GmailSendMessage
from langchain_community.tools.gmail.utils import build_resource_service

import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",  # or models/gemini-1.5-pro
    temperature=0.1
)

instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)

vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path="dataset.csv", source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)



def send_mail(user_email: str):
    # Build Gmail API service from credentials.json and token.json (in same folder)
    service = build_resource_service(
        credentials_path="credentials.json",
        token_path="token.json"
    )

    gmail_tool = GmailSendMessage(api_resource=service)

    office_support_email = "xyz@gmail.com"  # replace with your office support email
    subject = "Customer Support Request"
    message = f"Please contact {user_email} for support."

    args = {
        "to": office_support_email,
        "subject": subject,
        "message": message,
    }

    result = gmail_tool.invoke(args)
    return result



# def get_qa_chain():
#     vectordb = FAISS.load_local(
#     folder_path=vectordb_file_path,
#     embeddings=instructor_embeddings,
#     allow_dangerous_deserialization=True
#     )
#     retriever = vectordb.as_retriever(score_threshold=0.7)

#     prompt_template = """Given the following context and a question, generate an answer based on this context .
# In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
# Please make the answer short not too long.

# CONTEXT: {context}

# QUESTION: {question}"""

#     PROMPT = PromptTemplate(
#         template=prompt_template, input_variables=["context", "question"]
#     )

#     chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT},
#     )

#     return chain


def get_qa_chain():
    vectordb = FAISS.load_local(
        folder_path=vectordb_file_path,
        embeddings=instructor_embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate a concise and relevant answer based primarily on the "response" section in the source document context.
If the question is a greeting (like "hi", "hello", "hey"), respond politely but briefly.
If the question is outside the scope of the context or no relevant information is found, politely say that you don't have the information but you are here to help.
Keep your answer short and to the point.
CONTEXT: {context}

QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    main_chain = RunnableSequence([
        parallel_chain,
        RunnableLambda(lambda inputs: PROMPT.format(**inputs)),
        RunnableLambda(lambda prompt: llm.invoke(prompt))
    ])

    def run_chain(query):
        # query should be a dict with 'question' key, e.g. {"question": "hello?"}
        return main_chain.invoke(query)

    return run_chain

# Usage example:
if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    response = chain({"question": "hello?"})
    print(response)





