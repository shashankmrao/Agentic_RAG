import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, Literal, Sequence, TypedDict
#from langchain_classic import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.messages import  SystemMessage,HumanMessage
from langchain_groq import ChatGroq
import streamlit as st
import os

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT_NAME"] = "Agentic RAG Project"


# Initialize the tool
tavily_tool = TavilySearch(
    max_results=5
)

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm=ChatGroq(model_name="llama-3.3-70b-versatile")

urls=["https://breathedreamgo.com/india-travel-guide/"]
docs=[WebBaseLoader(url).load() for url in urls]
docs_list=[item for sublist in docs for item in sublist]
text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100,chunk_overlap=10)
texts=text_splitter.split_documents(docs_list)

vectorstore=Chroma.from_documents(
    documents=texts,
    collection_name="Indian_tourist_guide",
    embedding=embeddings)

bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 4. Combine into an Ensemble Retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7] # Balance between keyword and vector results
)

# Initialize the Ranker client first
flashrank_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

# Pass the client explicitly to bypass the validation bug
compressor = FlashrankRerank(client=flashrank_client, top_n=3)
# 2. Initialize the FlashRank compressor
#compressor = FlashrankRerank(top_n=3)

# 3. Create the Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=ensemble_retriever
)
retriever_tool = create_retriever_tool(
    name="Indian_tourist_Retriever",
    retriever=compression_retriever,
    description="A guide for Indian tourist information."
)
compression_retriever.invoke("Best places to visit in India?")

tools = [retriever_tool,tavily_tool]
tool_node=ToolNode(tools)
llm_with_tools=llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class grade(BaseModel):
    binary_score:str=Field(description="Relevance score 'yes' or 'no'")


def ai_assistant(state:AgentState):
    print("---CALL AGENT---")
    messages = state['messages']
    print(f"this is my message: {messages}")
    
    if len(messages)>1:
        print("---CALLING LLM---")
        response=llm.invoke(messages[-1].content)
        return {"messages": [response]}
    else:
        print("---CALLING LLM WITH TOOLS---")
        messages=[SystemMessage(content="""You are a helpful AI travel guide whose goal is to answer the user's question based on the following conditions:
        1. If the user's question is about travel in India or Indian tourism, use the retriever tool to get relevant context to answer the question.
        2. If the user's question is about current information or data, use the Tavily Search tool to get the latest information.
        3. If the user's question is about something else, use your general knowledge to answer.
        Provide a summarized output not more than 100 words.State your inability to answer if the question is not clear or if you don't have enough information. Do not provide inaccurate answers."""),
        HumanMessage(content=messages[0].content)]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

def grade_documents(state:AgentState)->Literal["Output_Generator", "Query_Rewriter"]:
    llm_with_structure_op=llm.with_structured_output(grade)
    
    prompt=PromptTemplate(
        template="""You are a grader deciding if the data or context is relevant to a user’s question.
                    Here is the data or context: {context}
                    Here is the user’s question: {question}
                    If the data or context talks about or contains information related to the user’s question, mark it as relevant. 
                    Give a 'yes' or 'no' answer to show if the information is relevant to the question.""",
                    input_variables=["context", "question"]
                    )
    chain = prompt | llm_with_structure_op
    
    messages = state["messages"]
    print(f"message for the grader: {messages}")
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: CONTEXT RELEVANT---")
        return "generator" #this should be a node name
    else:
        print("---DECISION: CONTEXT NOT RELEVANT---")
        return "rewriter" #this should be a node name
    
def generate(state:AgentState):
    print("---GENERATE---")
    messages = state["messages"]
    
    print(f"here is message from generate: {messages}")
    
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    print(f"---CONTEXT---: {docs}")
    
    prompt=PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of context or information to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context or Information: {context} 
    Answer:""",
    input_variables=["context", "question"]
                    )
    
    final_chain = prompt | llm

    response = final_chain.invoke({"context": docs, "question": question})
    print(f"this is my response:{response}")
    
    return {"messages": [response]}


def rewrite(state:AgentState):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    
    print(f"here is message from rewrite: {messages}")
    
    message = [HumanMessage(content=f"""Look at the input and try to reason about the underlying semantic intent or meaning. 
                    Here is the initial question: {question} 
                    Formulate an improved question: """)
       ]
    response = llm.invoke(message)
    print(f"this is my rewritten query: {response}")
    return {"messages": [response]}

workflow=StateGraph(AgentState)
workflow.add_node("My_AI_Assistant",ai_assistant)
workflow.add_node("tools", tool_node) 
workflow.add_node("Query_Rewriter", rewrite) 
workflow.add_node("Output_Generator", generate)

workflow.add_edge(START,"My_AI_Assistant")
workflow.add_conditional_edges("My_AI_Assistant",
                               tools_condition,
                               {"tools": "tools",
                                END: END,})

workflow.add_conditional_edges("tools",
                            grade_documents,
                            {"generator": "Output_Generator",
                            "rewriter": "Query_Rewriter"
                            }
                            )

workflow.add_edge("Output_Generator", END)
workflow.add_edge("Query_Rewriter", "My_AI_Assistant")
app=workflow.compile()



st.set_page_config(page_title="Agentic RAG Demo")
st.title("Agentic application")
st.header("Indian tourist guide")
input=st.text_input("Ask me anything about Indian tourism or current information about India!", key="user_input")
if st.button("Generate"):
    with st.spinner("Generating response..."):
        if input:
            response=app.invoke({"messages":[input]})
            st.subheader("Response:")
            st.write(response["messages"][-1].content)