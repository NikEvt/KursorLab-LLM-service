"""
API Module for Generating HTML Content and Structured Data

This module provides a FastAPI-based web service for generating HTML content and structured data
using a language model. It includes endpoints for generating styled HTML and iterative content
generation with streaming support.

Dependencies:
- dotenv: For loading environment variables from a `.env` file.
- langchain_community, langchain_core, langgraph: For interacting with the language model and managing prompts.
- fastapi: For building the web API.
- pydantic: For request validation.
- uvicorn: For running the FastAPI application.
- asyncio, multiprocessing, logging, nest_asyncio: For asynchronous and logging support.

Classes:
- StyleRequest: Pydantic model for validating the `style` parameter in the request body.

Functions:
- extract_html_from_ai_response: Extracts HTML code from the language model's response.
- generate_content_iterative: Generates content iteratively using a stack-based approach.

Endpoints:
- /generate_style/ (POST): Generates HTML content based on the provided style.
- /generate_content/ (POST): Generates structured content iteratively, with optional streaming support.

Main Execution:
- Starts a Uvicorn server to host the FastAPI application.
"""

import os
from typing import Optional, List, Dict, Union
from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb
from langchain_core.prompts import PromptTemplate
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import asyncio
import multiprocessing
import logging
import uvicorn
import nest_asyncio

import llm_functions as llm_func
from chroma_utils import ChromaTextFilesManager
import chromadb as cd
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingServer

from urllib.parse import urlparse
#### LOGGER INIT
"""
Initializes the logger for the application. Logs are written to `app.log` with a specific format.
"""
logging.basicConfig(
    level=logging.INFO,           # Minimum log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#### LLM INIT
try:
    model = ChatGoogleGenerativeAI(temperature=0.1,model='models/gemini-2.0-flash')
except Exception as e:
    logger.error("Error in LLM initialization: %s", str(e))
    raise e

#### CHROMA INIT

client_url = os.getenv("CHROMA_SERVER_URL")
if not client_url:
    raise ValueError("CHROMA_SERVER_URL is not set in the environment variables")
embedding_url = os.getenv("EMBEDDING_SERVER_URL")
if not embedding_url:
    raise ValueError("EMBEDDING_SERVER_URL is not set in the environment variables")

logger.info("CHROMA_SERVER_URL: %s", client_url)
logger.info("EMBEDDING_SERVER_URL: %s", embedding_url)

import re
parsed_url = urlparse(client_url)
host_from_url = parsed_url.hostname
port_from_url = parsed_url.port
logger.info("Chroma HOST: %s", host_from_url)
logger.info("Chroma PORT: %s", port_from_url)

client = cd.HttpClient(host=host_from_url, port=port_from_url)
embedding_function = HuggingFaceEmbeddingServer(url=embedding_url)
manager = ChromaTextFilesManager(client, embedding_function=embedding_function)

#### FASTAPI INIT
"""
Initializes the FastAPI application and defines the API endpoints.
"""
app = FastAPI()

class StyleRequest(BaseModel):
    """
    Pydantic model for validating the `style` parameter in the request body.
    """
    style: str

class AddTextRequest(BaseModel):
    """
    Pydantic model for validating the add_text request body.
    """
    text: str
    user_id: str
    doc_id: str
    # source name is optional
    source_name: Optional[str] = None
    # additional metadata is optional
    additional_metadata: Optional[dict] = None

class SearchParams(BaseModel):
    user_id: str
    doc_id: Optional[Union[str, List[str]]] = None

class ContentRequest(BaseModel):
    """
    Pydantic model for validating the content structure in the request body.

    If search_params is not provided, the content will be generated without searching for 
    relevant information in the database.
    """
    content: str
    html_code: str
    search_params: Optional[SearchParams] = None
    
class DeleteTextRequest(BaseModel):
    """
    Pydantic model for validating the delete_text request body.
    """
    user_id: str
    doc_id: str

@app.post("/add_text/")
async def add_text(request: AddTextRequest):
    """
    Adds text to the ChromaDB collection.
    """
    text = request.text
    user_id = request.user_id
    doc_id = request.doc_id
    source_name = request.source_name
    additional_metadata = request.additional_metadata
    try:
        manager.add_text(text, user_id, doc_id, source_name, additional_metadata)
        return {"message": "Text added to ChromaDB successfully"}
    except Exception as e:
        logger.error("Error in add_text: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

#delete text
@app.post("/delete_text/")
async def delete_text(request: DeleteTextRequest):
    """
    Deletes text from the ChromaDB collection.
    """
    user_id = request.user_id
    doc_id = request.doc_id
    try:
        manager.delete_text(user_id, doc_id)
        return {"message": "Text deleted from ChromaDB successfully"}
    except Exception as e:
        logger.error("Error in delete_text: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_style/")
async def generate_style(request: StyleRequest):
    """
    Generates HTML content based on the provided style.

    Args:
        request (StyleRequest): The request body containing the `style` parameter.

    Returns:
        dict: A dictionary containing the generated HTML code or an error message.
    """
    style = request.style
    logger.info("Received generate_style request with style: %s", style)
    try:
        improved_prompt = llm_func.convert_style(model, style)
        logger.info("Improved prompt generated successfully")
        html_code = llm_func.generate_lesson_design(model, improved_prompt)
        logger.info("HTML code generated successfully")
        html_code = llm_func.extract_code_from_ai_response(html_code)
        logger.info("HTML code extracted successfully")
        logger.info("Generated HTML code: %s", html_code)
        return {"html_code": html_code}
    except Exception as e:
        logger.error("Error in generate_style: %s", str(e))
        return {"error": str(e)}

@app.post("/generate_content/")
async def generate_content(content: ContentRequest):
    """
    Generates structured content iteratively, with optional streaming support.

    Args:
        content (dict): The content structure to process.
        stream (bool): Whether to enable streaming response.

    Returns:
        StreamingResponse or list: The generated content as a stream or a full response.
    """
    if content.content is None:
        raise HTTPException(status_code=400, detail="Content is required")
    if content.html_code is None:
        raise HTTPException(status_code=400, detail="HTML code is required")
    if content.search_params is not None:
        if content.search_params.user_id is None:
            raise HTTPException(status_code=400, detail="User ID is required")
        my_manager = manager
    else:
        my_manager = None
    logger.info("Received generate_content request with content: %s", content)
    try:
        content_request = content.content
        html_template = content.html_code
        logger.info("Content request: %s", content_request)
        logger.info("HTML template: %s", html_template)
        relevant_info_from_files = None
        if my_manager is not None:
            relevant_info_from_files = llm_func.simple_search_info_for_lesson(model,content_request,my_manager, content.search_params.user_id, content.search_params.doc_id)
        logger.info("Relevant info from files: %s", relevant_info_from_files)
        lesson_plan = llm_func.generate_lesson_plan(model, content_request, html_template, relevant_info_from_files)
        logger.info("Lesson plan generated successfully")
        logger.info("Lesson plan: %s", lesson_plan)
        lesson = llm_func.improve_text(model, content_request,lesson_plan, content.search_params.user_id, doc_id=content.search_params.doc_id, text_files_manager=my_manager)
        logger.info("Lesson improved successfully")
        lesson = llm_func.fill_html_with_text(html_template, lesson)
        lesson = llm_func.extract_code_from_ai_response(lesson, code_lang='html')
        logger.info("Lesson extracted successfully")
        return {"lesson": lesson}
    except Exception as e:
        logger.error("Error in generate_content: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    """
    Starts the Uvicorn server to host the FastAPI application.
    """
    nest_asyncio.apply()
    logger.info("Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
