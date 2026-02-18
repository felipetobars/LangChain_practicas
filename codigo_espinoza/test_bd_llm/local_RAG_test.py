import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import Tool
from dotenv import load_dotenv
load_dotenv()

DB_DIR = "./chroma_db_hp"
PDF_PATH = 'HP7.pdf'

#embeddings = OllamaEmbeddings(model="nomic-embed-text")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

if os.path.exists(DB_DIR):
    print("--- Cargando Base de Datos existente ---")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    print("--- Procesando PDF ---")
    loader = PyPDFLoader(PDF_PATH)
    data = loader.load()

    def clean_text(text):
        text = re.sub(r'\n\d+\s*\n', '\n', text)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        return " ".join(text.split())

    data_filtered = [doc for doc in data if len(doc.page_content.strip()) > 10]
    for doc in data_filtered:
        doc.page_content = clean_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,        
        chunk_overlap=700,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(data_filtered)

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_DIR
    )
    print(f"--- DB creada con {len(chunks)} chunks ---")

# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.7}
# )

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# LLM definido ANTES de search_hp_book para poder usarlo en HyDE
llm = ChatOllama(model="qwen2.5:7b", temperature=0.5) # 
#llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-2.5-flash')

def search_hp_book(query: str) -> str:
    # HyDE correcto: usa el query real del usuario
    hyde_prompt = f"Escribe un fragmento breve de Harry Potter 7 en español que responda: {query}"
    try:
        hyde_response = llm.invoke(hyde_prompt).content
    except Exception:
        hyde_response = ""

    # Extrae nombres propios del query para reforzar la búsqueda
    keywords = " ".join(re.findall(r'[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+', query))
    
    combined_query = f"{query} {hyde_response} {keywords}".strip()

    docs = retriever.invoke(combined_query)
    
    if not docs:
        return "No se encontraron fragmentos relevantes."
    
    return "\n\n".join([
        f"[Pág {d.metadata.get('page', '?')}]: {d.page_content}" 
        for d in docs
    ])

hp_tool = Tool(
    name="search_harry_potter",
    func=search_hp_book,
    description=(
        "Busca fragmentos del libro Harry Potter 7 en español. "
        "Úsalo con términos variados si el primer intento falla. "
        "NO debes usar siempre como entrada palabras literales (usa sinónimos o conceptos)."
        "Debes corregir nombres de personajes si se ingresan mal y"
    )
)

template = """Eres un experto en Harry Potter 7 (Las Reliquias de la Muerte).

Tienes acceso a estas herramientas:
{tools}

INSTRUCCIONES:
- Si no encuentras la respuesta, intenta con DIFERENTES términos de búsqueda.
- Varía los queries: usa el lugar de la escena, objetos, o acciones alternativas.
- Nunca repitas el mismo Action Input dos veces.
- Máximo {max_iterations} intentos.

Formato OBLIGATORIO:

Question: {input}
Thought: [tu razonamiento]
Action: {tool_names}
Action Input: [términos de búsqueda específicos y variados]
Observation: [resultado]
... (repite con queries diferentes si es necesario)
Thought: [conclusión]
Final Answer: [respuesta detallada en español, o indica que no se encontró]

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm, [hp_tool], prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[hp_tool],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=6
)

def preguntar(pregunta: str):
    print(f"\nUsuario: {pregunta}")
    try:
        response = agent_executor.invoke({"input": pregunta, "max_iterations": 6})
        print(f"\nRespuesta Final:\n{response['output']}")
    except Exception as e:
        print(f"\nError: {e}")
    print("\n" + "="*50)

if __name__ == "__main__":
    while True:
        question = input("Haz tu pregunta: ")
        preguntar(question)
        if input("¿Otra pregunta? (s/n): ").lower() == 'n':
            break