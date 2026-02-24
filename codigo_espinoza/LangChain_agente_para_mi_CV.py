import os, re, asyncio, sys
from dotenv import load_dotenv
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.theme import Theme

os.environ["USER_AGENT"] = "my-app/1.0"

custom_theme = Theme({
    "markdown.h1": "bold magenta",
    "markdown.h2": "bold cyan",
    "markdown.h3": "bold blue",
    "markdown.bold": "bold yellow",
    "markdown.italic": "italic green",
    "markdown.code": "bold red on grey15",
    "markdown.code_block": "grey82 on grey15",
    "markdown.list": "grey74",
    "markdown.item.bullet": "cyan",
    "markdown.link": "bright_blue underline",
})
console = Console(theme=custom_theme, force_terminal=True)

# ---- LOADER WEB ----
async def load_web_docs(urls: list[str]):
    loader = AsyncChromiumLoader(urls)
    raw_docs = await loader.aload()
    bs_transformer = BeautifulSoupTransformer()
    return bs_transformer.transform_documents(
        raw_docs,
        tags_to_extract=["p", "h1", "h2", "h3", "span", "li"]
    )

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

def clean_docs(docs):
    for doc in docs:
        # Eliminar URLs largas
        doc.page_content = re.sub(r'https?://\S+', '', doc.page_content)
        # Eliminar espacios y saltos de línea excesivos
        doc.page_content = re.sub(r'\n{3,}', '\n\n', doc.page_content)
        doc.page_content = re.sub(r' {2,}', ' ', doc.page_content)
    return docs

pags = asyncio.run(load_web_docs(["https://felipetobars.github.io/lfts_portfolio/"]))

pags = clean_docs(pags)
# ---- PIPELINE ----
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
docs = text_splitter.split_documents(pags)


embeddings = OllamaEmbeddings(model="paraphrase-multilingual:latest")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name="portfolio-web")
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 10, "lambda_mult": 0.7})

# ---- PROMPT CON HISTORIAL ----
prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente de CV de Felipe para otras personas, experto en responder preguntas basándote únicamente en el contexto proporcionado.
Responde siempre en el mismo idioma en que esté escrita la pregunta.
Usa formato Markdown rico en tus respuestas:
- **Negrita** para conceptos importantes
- Encabezados ## y ### para estructurar
- Listas cuando corresponda
- `código` para términos técnicos
- Tablas cuando compares información

Contexto del documento:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

load_dotenv()
# llm = ChatOllama(model="qwen2.5:7b", temperature=0, keep_alive=-1)
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite', google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.7) 

# ---- HISTORIAL ----
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain = prompt | llm | StrOutputParser()

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# ---- RAG ----
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question: str, session_id: str = "default"):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)

    # # DEBUG - agregar esto temporalmente
    # console.print("\n[bold red]--- CONTEXTO RECUPERADO ---[/bold red]")
    # console.print(formatted_context)
    # console.print("[bold red]--- FIN CONTEXTO ---[/bold red]\n")

    full_response = ""
    with Live(console=console, refresh_per_second=10) as live:
        for chunk in conversation.stream(
            {"question": question, "context": formatted_context},
            config={"configurable": {"session_id": session_id}}
        ):
            full_response += chunk
            live.update(Markdown(full_response))

    return full_response

# ---- LOOP ----
while True:
    question = input("\nHaz tu pregunta: ")
    rag_chain(question)

    again = input("\n¿Deseas hacer otra pregunta? (s/n): ").strip().lower()
    if again == "n":
        break