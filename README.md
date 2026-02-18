# Prácticas de LangChain: Entorno Local y Colab

Este repositorio contiene prácticas y ejemplos para aprender a usar LangChain, una librería para construir aplicaciones de inteligencia artificial que integran modelos de lenguaje, bases de datos vectoriales y otras herramientas.

## Descripción

Las prácticas exploran conceptos como:
- Uso de modelos de lenguaje locales (Ollama)
- Integración con bases de datos vectoriales (ChromaDB)
- Recuperación aumentada por generación (RAG)
- Uso de LangChain en entornos locales y en Google Colab

**Nota:** El repositorio está en desarrollo y algunas prácticas pueden estar incompletas. El README se actualizará conforme se agreguen nuevas instrucciones y archivos.

## Preparación del entorno local

### 1. Instalar Ollama
Ollama permite ejecutar modelos de lenguaje localmente. Descárgalo e instálalo desde [https://ollama.com](https://ollama.com).

### 2. Crear entorno Conda
Usa Conda para gestionar el entorno Python:

```bash
conda create -n langchain python=3.12 -y
conda activate langchain
```

### 3. Instalar PyTorch (con soporte CUDA)

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 4. Instalar dependencias principales

```bash
pip3 install langchain langchain-community pypdf openai chromadb tiktoken sentence-transformers transformers ipykernel langchain-google-genai langchain_ollama langchain_experimental duckduckgo-search ddgs wikipedia pinecone langchain-pinecone
```

## Preparación en Google Colab

En Colab, la instalación de librerías frameworks es similar (hay algunos cambios respecto a como usar Ollama+GPU T4 que se epxlican en el notebook):

```python
!pip install langchain langchain-community pypdf openai chromadb tiktoken sentence-transformers transformers ipykernel langchain-google-genai langchain_ollama langchain_experimental duckduckgo-search ddgs wikipedia pinecone langchain-pinecone
```

**Nota:** Algunas funcionalidades (como Ollama local) no están disponibles en Colab.

## Actualización

Este README se irá actualizando conforme se completen las prácticas y se agreguen nuevas instrucciones.

---


