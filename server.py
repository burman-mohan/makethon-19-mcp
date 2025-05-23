"""
MCP (Machine-to-Machine Communication Protocol) server script for
code parsing and vector database searching.

This server provides two main tools:
1. Code Parser Tool: Parses source code from a directory and stores embeddings in Qdrant
2. Qdrant Query Tool: Performs semantic search on stored code embeddings

Key functionalities:
- Parses and processes source code files from specified directories
- Generates and stores code embeddings in Qdrant vector database
- Performs semantic similarity search on stored code embeddings
- Configurable via environment variables for database connection and server settings
"""

from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
import json
import os

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

# --- Configuration ---
DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code" # Default model if not set in .env
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8050
DEFAULT_TRANSPORT = "sse"

# Determine effective host and port to be used by the server and for logging
EFFECTIVE_HOST = os.getenv("HOST", DEFAULT_HOST)
EFFECTIVE_PORT = int(os.getenv("PORT", str(DEFAULT_PORT)))


@dataclass
class AppContext:
    """
    Application context holding shared resources for the MCP server.

    Attributes:
        qdrant_client: An initialized QdrantClient instance.
        embedding_model: An initialized HuggingFaceEmbeddings instance for generating query vectors.
    """
    qdrant_client: QdrantClient
    embedding_model: HuggingFaceEmbeddings


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Manages the lifecycle of application clients (Qdrant client, embedding model).

    This context manager initializes resources when the server starts and
    ensures they are cleaned up when the server stops.

    Args:
        server: The FastMCP server instance (not directly used in this example,
                but available for more complex lifespan logic).

    Yields:
        AppContext: The context containing the initialized Qdrant client and embedding model.
    """
    # Initialize Qdrant Client
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        print("Error: QDRANT_URL environment variable not set. Qdrant client cannot be initialized.")
        raise ValueError("QDRANT_URL environment variable is required.")

    print(f"Attempting to connect to Qdrant at {qdrant_url}...")
    qdrant_client = QdrantClient(url=qdrant_url)
    try:
        qdrant_client.get_collections() # A simple check to verify connection and permissions
        print("Successfully connected to Qdrant and listed collections.")
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant or list collections: {e}. "
              "The 'query_qdrant' tool might not function correctly.")

    # Initialize Embedding Model
    embedding_model_name = os.getenv("SENTENCE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    print(f"Loading sentence embedding model: {embedding_model_name}...")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print(f"Embedding model '{embedding_model_name}' loaded successfully.")
    except Exception as e:
        print(f"Error: Failed to load embedding model '{embedding_model_name}': {e}")
        raise

    app_context = AppContext(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model
    )

    try:
        yield app_context
    finally:
        # Cleanup resources
        if hasattr(qdrant_client, 'close'):
            qdrant_client.close()
            print("Qdrant client closed.")
        print("Application lifecycle ended. Resources cleaned up.")


# Initialize FastMCP server
mcp_server = FastMCP(
    service_id="mcp-qdrant-searcher",
    description="MCP server providing tools for Qdrant vector database interaction.",
    lifespan=app_lifespan,
    host=EFFECTIVE_HOST, # Use the determined host
    port=EFFECTIVE_PORT  # Use the determined port
)

@mcp_server.tool()
async def calculate_days(startDate: str, endDate:str) -> int:
    """
    Calculate number of days.

    Args:
        startDate: Start date
        endDate: End date


    Returns:
        str: Number of days between start date and end date
    """

    print("startDate: ", startDate)
    print("endDate: ", endDate)

    return 10

@mcp_server.tool()
async def parse_code(directory_path: str, language: str) -> str:
    """
    Use this tool to parse code

    Args:
        directory_path: Path of the directory
        language: Programming language


    Returns:
        str: Collection name
    """

    print("directory_path: ", directory_path)
    print("language1: ", language)

    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Error Directory '{directory_path}' does not exist.")
        return json.dumps({"error": f"Directory '{directory_path}' does not exist."})
    if not os.path.isdir(directory_path):
        print(f"Error Path '{directory_path}' is not a directory.")
        return json.dumps({"error": f"Path '{directory_path}' is not a directory."})
    print(f"Parsing code files from directory: {directory_path}")
    print(f"Language specified: {language}")


    # Load documents with LanguageParser first
    loader = GenericLoader.from_filesystem(
        directory_path,
        suffixes=[".java"],
        parser=LanguageParser(language="java")
    )

    # Load and parse the documents
    documents = loader.load()
    print(f"Number of loaded documents: {len(documents)}")

    # Combine all documents' content into a single string
    all_documents = ""
    for doc in documents:
        all_documents += f"--- Document Metadata ---\n{doc.metadata}\n"
        all_documents += f"--- Document Content ---\n{doc.page_content}\n"
        all_documents += "-" * 80 + "\n"  # Separator between documents

    # #Print the first document's content
    # for doc in documents:
    #     print(f"Document content: {doc.page_content}")
    #     print(f"Document metadata: {doc.metadata}")
    #     break  # Print only the first document
    print(all_documents)

    return all_documents


# @mcp_server.tool()
# async def parse_code(
#     directory_url: str,
#     language: str,
# ) -> str:
#
#     """
#     Ingest code files from a directory.
#
#     Args:
#         directory_url: Path to the directory containing source code files
#         language: Programming language of the source files (e.g., 'js', 'python')
#
#     Returns:
#         str: Name of the created Qdrant collection containing the code embeddings
#     """
#
#     # Check if the directory exists
#     if not os.path.exists(directory_url):
#         return json.dumps({"error": f"Directory '{directory_url}' does not exist."})
#     if not os.path.isdir(directory_url):
#         return json.dumps({"error": f"Path '{directory_url}' is not a directory."})
#     print(f"Parsing code files from directory: {directory_url}")
#     print(f"Language specified: {language}")
#
#
#     # Load documents with LanguageParser first
#     loader = GenericLoader.from_filesystem(
#         "D:\Codes\Python\makethon-19\React_E-Commerce-main",
#         suffixes=[".js"],
#         parser=LanguageParser(language="js")
#     )
#
#     # Load and parse the documents
#     documents = loader.load()
#     print(f"Number of loaded documents: {len(documents)}")
#
#     # Print the first document's content
#     # for doc in documents:
#     #     print(f"Document content: {doc.page_content}")
#     #     print(f"Document metadata: {doc.metadata}")
#     #     break  # Print only the first document
#
#     # Further split with language-aware text splitter
#     js_splitter = RecursiveCharacterTextSplitter.from_language(
#         language=Language.JS,
#         chunk_size=500,
#         chunk_overlap=0
#     )
#     split_docs = js_splitter.split_documents(documents)
#
#     # Print the number of split documents
#     print(f"Number of split documents: {len(split_docs)}")
#     # Print the first 20 split document's content
#     for i, doc in enumerate(split_docs[:20]):
#         print(f"Split document {i + 1} content: {doc.page_content}")
#         print(f"Split document {i + 1} metadata: {doc.metadata}")
#     # for doc in split_docs:
#     #     print(f"Split document content: {doc.page_content}")
#     #     print(f"Split document metadata: {doc.metadata}")
#     #     break  # Print only the first split document
#
#     # Initialize the embedding model
#     embedding_model = HuggingFaceEmbeddings(
#         model_name="jinaai/jina-embeddings-v2-base-code"
#     )
#
#     # Create embeddings for the split documents
#     embeddings = embedding_model.embed_documents([doc.page_content for doc in split_docs])
#
#     print(f"Generated embeddings for {len(embeddings)} documents.")
#
#     # Connect to local Qdrant instance
#     qdrant = QdrantClient(host="192.168.1.100", port=6333)
#
#     collection_name = "js_code_embeddings"
#
#     # Create collection if it doesn't exist
#     if collection_name not in [c.name for c in qdrant.get_collections().collections]:
#         qdrant.create_collection(
#             collection_name=collection_name,
#             vectors_config=VectorParams(
#                 size=len(embeddings[0]),
#                 distance=Distance.COSINE
#             )
#         )
#
#     # Prepare points for upsert
#     points = [
#         PointStruct(
#             id=i,
#             vector=embeddings[i],
#             payload={
#                 "content": split_docs[i].page_content,
#                 "metadata": split_docs[i].metadata
#             }
#         )
#         for i in range(len(embeddings))
#     ]
#
#     # Upsert points into Qdrant
#     qdrant.upsert(
#         collection_name=collection_name,
#         points=points
#     )
#
#     print(f"Saved {len(points)} embeddings to Qdrant collection '{collection_name}'.")
#
#     return collection_name




@mcp_server.tool()
async def query_qdrant(
    ctx: Context,
    collection_name: str,
    query_text: str,
    top_k: int = 3
) -> str:
    """
    Search for semantically similar code snippets in the stored embeddings.

    Args:
        ctx: The MCP server context
        collection_name: Name of the Qdrant collection containing code embeddings
        query_text: Natural language description of the code you're looking for
        top_k: Number of most similar code snippets to return (default: 3)

    Returns:
        str: JSON string containing the most similar code snippets, including:
            - id: Unique identifier of the code snippet
            - score: Similarity score (higher is better)
            - payload: Contains the actual code content and metadata
    """
    
    try:
        app_resources: AppContext = ctx.request_context.lifespan_context
        qdrant_client = app_resources.qdrant_client
        embedding_model = app_resources.embedding_model

        if not qdrant_client:
            return json.dumps({"error": "Qdrant client not available."})
        if not embedding_model:
            return json.dumps({"error": "Embedding model not available."})

        print(f"Generating embedding for query: '{query_text}'")
        query_vector = embedding_model.embed_query(query_text)

        print(f"Searching collection '{collection_name}' in Qdrant for top {top_k} results.")
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        formatted_results = []
        for hit in search_results:
            formatted_results.append({
                "id": str(hit.id),
                "score": float(hit.score),
                "payload": hit.payload,
            })

        print(f"Found {len(formatted_results)} results.")
        return json.dumps(formatted_results, indent=2)

    except models.UnexpectedResponse as qdrant_api_ex:
        if qdrant_api_ex.status_code == 404 or \
           ("collection" in str(qdrant_api_ex).lower() and "not found" in str(qdrant_api_ex).lower()):
            error_message = f"Error: Qdrant collection '{collection_name}' not found."
            print(error_message)
            return json.dumps({"error": error_message, "details": str(qdrant_api_ex)})
        else:
            error_message = f"Qdrant API error: {str(qdrant_api_ex)}"
            print(error_message)
            return json.dumps({"error": error_message})
    except Exception as e:
        error_message = f"An unexpected error occurred while querying Qdrant: {str(e)}"
        print(f"{error_message} - Type: {type(e)}")
        return json.dumps({"error": error_message})


async def main():
    await parse_code(r"D:\Codes\Python\makethon-19\final_code\agentFwk\res_code\myFolder-compressed", "java")
    """
    Main asynchronous function to start the MCP server.
    Selects the transport (SSE or STDIO) based on the TRANSPORT environment variable.
    """
    transport_mode = os.getenv("TRANSPORT", DEFAULT_TRANSPORT).lower()

    # Use the globally determined EFFECTIVE_HOST and EFFECTIVE_PORT for printing
    if transport_mode == 'sse':
        print(f"Starting MCP server with SSE transport on http://{EFFECTIVE_HOST}:{EFFECTIVE_PORT}")
        await mcp_server.run_sse_async()
    elif transport_mode == 'stdio':
        print("Starting MCP server with STDIO transport.")
        await mcp_server.run_stdio_async()
    else:
        print(f"Warning: Unknown transport mode '{transport_mode}'. Defaulting to SSE.")
        print(f"Starting MCP server with SSE transport on http://{EFFECTIVE_HOST}:{EFFECTIVE_PORT}")
        await mcp_server.run_sse_async()


if __name__ == "__main__":
    print("Initializing MCP server for Qdrant...")
    asyncio.run(main())