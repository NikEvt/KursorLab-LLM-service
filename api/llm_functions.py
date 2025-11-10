"""
This module provides functions for working with LLM (Large Language Model) based educational content generation.
It includes utilities for HTML processing, lesson plan generation, and content improvement.

The module is designed to work with educational content, allowing for:
- HTML structure extraction and manipulation
- Lesson plan generation and improvement
- Style conversion and template management
- Text improvement and formatting

Dependencies:
- langchain_core: For LLM interactions and prompt templates
- BeautifulSoup: For HTML parsing and manipulation
- yaml: For YAML serialization/deserialization
- re: For regular expression operations
"""

from langchain_core.prompts import PromptTemplate,PipelinePromptTemplate
from copy import deepcopy
from langchain_core.language_models.llms import BaseLLM
import yaml
from bs4 import BeautifulSoup
import codecs
import html
import re
from chroma_utils import ChromaTextFilesManager
from typing import Optional, Union, List, Any, Dict

GENERAL_PIPELINE_TEMPLATE = PromptTemplate.from_template(
"""
#Context
{context}

#Task
**Role**
{role}

**Constraints**
{constraints}

**Output**
"""
)

def simple_search_info_for_lesson(model: BaseLLM, 
                           lesson_theme: str, 
                           text_files_manager: ChromaTextFilesManager, 
                           user_id: str, 
                           doc_id: Optional[Union[str, List[str]]] = None,
                           n_results: int = 10,
                           where: Optional[Dict[str, Any]] = None) -> str:
    """
    Searches for relevant information in text files for a given lesson theme.
    
    Args:
        model (BaseLLM): The language model to use for search
        lesson_theme (str): The theme of the lesson
        text_files_manager (ChromaTextFilesManager): The text files manager to use for search
        user_id (str): ID of the user to search for
        doc_id (Optional[Union[str, List[str]]]): Specific document ID(s) to search within
        n_results (int): Number of results to return
        where (Optional[Dict[str, Any]]): Additional where conditions for filtering
        
    Returns:
        str: A YAML-structured lesson plan
        
    Note:
        - Uses the language of the request
        - Follows specified template structure
        - Includes task formatting
    """
    search_results = text_files_manager.search_documents(
        query=lesson_theme,
        user_id=user_id,
        n_results=n_results,
        doc_id=doc_id,
        where=where
    )
    
    # Escape curly braces in the content to prevent template variable interpretation
    def escape_curly_braces(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")
    
    context = "\n\n".join([
        f"Content: {escape_curly_braces(result['content'])}\nSource: {result['metadata'].get('source', 'Unknown')}"
        for result in search_results
    ])
    return context
 
def extract_html_structure(html_content: str) -> dict:
    """
    Extracts the structure of an HTML document into a nested dictionary format.
    
    Args:
        html_content (str): The HTML content to parse
        
    Returns:
        dict: A nested dictionary representing the HTML structure, where:
            - Keys are HTML tag names
            - Values are dictionaries containing:
                - 'id': Element ID if present
                - 'text': Text content for text-containing elements
                - 'children': List of child elements (recursive structure)
                
    Note:
        - Ignores 'b' and 'i' tags
        - Preserves text content for headings and paragraphs
        - Maintains element hierarchy
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.body

    def traverse_and_extract(element, level=0):
        structure = {}
        tag_name = element.name
        structure[tag_name] = {}

        # Extract id if has
        if element.has_attr('id'):
            structure[tag_name]['id'] = element['id']

        # Recursively traverse children
        children = list(element.children)
        # Filter out text children
        children = [child for child in children if not isinstance(child, str)]
        #filter out b,i tags
        children = [child for child in children if child.name not in ['b','i']]

        flag=False
        if children:
          structure[tag_name]['children'] = []
          for child in children:
              if child.name: # and child.name not in ['i']:  # Check if child is a tag
                  structure[tag_name]['children'].append(traverse_and_extract(child, level+1))

        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] or element.name == 'p':
          structure[tag_name]['text'] = ""
          for i in element.contents:
            if i.name not in ['i']:
              structure[tag_name]['text'] += i.get_text(strip=True)
          #structure[tag_name]['text'] = element.get_text(strip=True)
        elif not children:
          text = element.get_text(strip=True)
          if text:
            structure[tag_name]['text'] = text

        return structure

    return traverse_and_extract(body)

def extract_code_from_ai_response(response_text: str, code_lang: str = 'html') -> str:
    """
    Extracts code blocks from AI response text.
    
    Args:
        response_text (str): The text response from the AI
        code_lang (str, optional): The language of the code block to extract. Defaults to 'html'.
        
    Returns:
        str: The extracted code block if found, otherwise the original response text
    """
    pattern = r"```" + code_lang +r"\n(.*?)\n```"
    match = re.search(pattern, response_text, re.DOTALL)  # re.DOTALL to match across newlines

    if match:
        return match.group(1).strip()  # Return the captured group (the code), and remove leading/trailing whitespace
    else:
        return response_text
    
def fill_html_with_text(html_content: str, lesson_plan_dict: dict) -> str:
    """
    Fills an HTML template with content from a lesson plan dictionary.
    
    Args:
        html_content (str): The HTML template to fill
        lesson_plan_dict (dict): Dictionary containing the lesson plan content
        
    Returns:
        str: The filled HTML content with text inserted at appropriate locations
        
    Note:
        - Preserves HTML structure while replacing text content
        - Uses element IDs to match content with locations
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    def traverse_struct(struct):
        tag_name, struct = list(struct.items())[0]
        if 'children' in struct:
            for child in struct['children']:
                traverse_struct(child)
        if 'id' in struct and 'text' in struct:
            element = soup.find(id=struct['id'])
            if element:
                for i in element.contents:
                  if i.name not in ['i']:
                    i.extract()
                element.append(struct['text'])
                    

    traverse_struct(lesson_plan_dict)
    return html.unescape(str(soup))

def convert_style(model: BaseLLM, style: str) -> str:
    """
    Converts a style description into a formatted lesson style specification.
    
    Args:
        model (BaseLLM): The language model to use for conversion
        style (str): The style description to convert
        
    Returns:
        str: A formatted style specification including:
            - Font selection
            - Color scheme
            - Layout structure
            - Text formatting rules
    """
    CONVERT_TEMPLATE = PromptTemplate.from_template('''Generate a lesson style in the format as in the example.
The style must contain a main heading.
You can add details to the style if you think it will improve the final result.
The background should usually not distract from the text, so do not choose bright colors for the background.
Usually, the color should differ very slightly from white or black.
YOU ARE NOT ALLOWED TO USE GRAY FOR PAGE BACKGROUND COLOR
YOU MUST USE LANGUAGE OF REQUEST
Please Select Fonts which are usable with language of request

If you think about choosing colors you can use following schema:
1) Select a base color
2) Choose very light(close to white) version of base color and close to black
3) choose complementary color for accent

Choose the light version for background, and dark for  text if it is not darkmode style

EXAMPLE:
Request:

Стиль: Шрифт Comforta, в теплых тонах, строгий стиль
Структура: Три параграфа с заголовками, во втором булет лист

Response:

общий стиль: Шрифт Comforta, цвет текста темно коричневый,  фон страницы светло бежевый
заголовок урока:
-заголовок 1:
--параграф:
-заголовок 2:
--параграф:
--буллет лист: Цвет текста красный
--параграф:
-заголовок 3:
--параграф:

**START EXAMPLE:**
Request:

Стиль: Шрифт Comforta, в теплых тонах, строгий стиль, третий параграф должен быть с красным текстом
Структура: Три параграфа с заголовками, во втором булет лист

Response:

общий стиль: Шрифт Comforta, цвет текста темно коричневый,  фон страницы светло бежевый
заголовок урока:
-заголовок 1:
--параграф:
-заголовок 2:
--параграф:
--буллет лист: Цвет текста красный
--параграф:
-заголовок 3:
--параграф:
**END EXAMPLE:**


**START EXAMPLE:**
Стиль: Шрифт строгий, но не обычный, строгий стиль, строгая цветовая гамма (чероно-белая)
Структура: Главный заголовок, три параграфа о уроке с подзаголовками, и секция с 3 заданиями

Response:

общий стиль: Шрифт Roboto, цвет текста черный, фон страницы очень светлый серый (почти белый)
главный заголовок:
-заголовок 1:
--параграф:
-заголовок 2:
--параграф:
-заголовок 3:
--параграф:
-секция заданий:
--задание 1:
--задание 2:
--задание 3:
**END EXAMPLE:**

TASK:
Request:
{style}
Response:
'''
)
    return model.invoke(CONVERT_TEMPLATE.format(style=style)).content

def generate_lesson_design(model: BaseLLM, lesson_content: str) -> str:
    """
    Generates an HTML lesson design based on content specifications.
    
    Args:
        model (BaseLLM): The language model to use for generation
        lesson_content (str): The lesson content specifications
        
    Returns:
        str: An HTML document containing the lesson design
        
    Note:
        - Includes unique IDs for all text elements
        - Maintains specified structure
        - Adds appropriate styling and icons
    """
    LESSON_DESIGN_TEMPLATE = PromptTemplate.from_template( """You are a highly skilled educational disigner tasked with generating a complete, interactive HTML lesson.

1. Please output a html file with the **style and structure**
2. Please fill any text fields by its simple name. For example h as title, the p as paragraph, etc. But keep in mind YOU MUST USE LANGUAGE OF **Lesson Specifications:** FOR ALL TEXT
3. DONT RETRUN ANYTHING EXCEPT WEB PAGE
4. Please keep structure from Lesson Specifications and dont add new elements to layout
5. It is highly prized to add some nice elements to the page even if them are not in specefiction. But keep in mind that you are not allowed to add any forms since this lesson will be launched without backend.
6. Also if you will break structure of a lesson than you will be punished
7. Add nice icons to highlit sections (for them use google icons for example or other open resources)
8. Add unique id to each text field (p,h1,h2,h3,h4,h5,h6, li and the ect.)
9. Tasks or assigments should have not only title but also description of what to do
**Lesson Specifications:**

{lesson_content}

**Output:**
""")
    return model.invoke(LESSON_DESIGN_TEMPLATE.format(lesson_content=lesson_content)).content

def generate_lesson_plan(model: BaseLLM, lesson_theme: str, html_template: str, relevant_info_from_files: Optional[str] = None) -> str:
    """
    Generates a lesson plan based on a theme and HTML template.
    
    Args:
        model (BaseLLM): The language model to use for generation
        lesson_theme (str): The theme of the lesson
        html_template (str): The HTML template to use as a base
        
    Returns:
        str: A YAML-structured lesson plan
        
    Note:
        - Uses the language of the request
        - Follows specified template structure
        - Includes task formatting
    """
    LESSON_CONTEXT = PromptTemplate.from_template('Lesson theme: {lesson_theme}\nStruct: {struct}')
    if relevant_info_from_files:
        LESSON_CONTEXT += f'\nRelevant info:\n\n{relevant_info_from_files}\n\n'
    LESSON_ROLE = PromptTemplate.from_template('You are a highly skilled educational write. You is supposed to write a lesson draft for **LESSON THEME**. Please output a yaml with **STRUCT** with replaced text with decription of what to write in THIS section. You may use info from **Relevant info** if it is relevant to the lesson. If you use info from **Relevant info** please dublicate part that you use from **Relevant info**, especially dublicate equations and other math expressions and if you want to add citation')
    LESSON_CONSTRAINTS = PromptTemplate.from_template('- You must use languge of request\n- If you write a task please write in format f\'Task:text of a task\'\n')
    LESSON_PLAN_TEMPLATE = PromptTemplate.from_template(
       GENERAL_PIPELINE_TEMPLATE.invoke(
        {'context':LESSON_CONTEXT.template,
         'role':LESSON_ROLE.template,
         'constraints':LESSON_CONSTRAINTS.template}
        ).text
    )
    structure = extract_html_structure(html_template)
    structure_yml = yaml.dump(structure, indent=2)
    response = model.invoke(LESSON_PLAN_TEMPLATE.format(lesson_theme=lesson_theme, struct=structure_yml)).content
    response = extract_code_from_ai_response(response,code_lang='yaml')
    response = yaml.safe_load(response)
    return response

def improve_text(model: BaseLLM, stakeholder_request: str, lesson_plan: dict, 
                 user_id: str, doc_id: Optional[Union[str, List[str]]] = None,
                 text_files_manager: Optional[ChromaTextFilesManager] = None) -> dict:
    """
    Improves lesson text based on stakeholder requirements and existing content.
    
    Args:
        model (BaseLLM): The language model to use for improvement
        stakeholder_request (str): Requirements from stakeholders
        lesson_plan (dict): The existing lesson plan to improve
        
    Returns:
        dict: An improved version of the lesson plan
        
    Note:
        - Maintains HTML formatting
        - Adds emphasis to important points
        - Improves exercises and explanations
        - Preserves language consistency
    """
    LESSON_ELEMENT_CONTEXT = PromptTemplate.from_template("""
**Stakeholder Request**
{stakeholder_request}

**Lesson plan**
{lesson_plan}

**Text written**
{text_written}

**What to write**
{what_to_write}
""")
    LESSON_ELEMENT_ROLE = PromptTemplate.from_template("""
You are a 1-response API. Your output must strictly be provided in html format.
You are a highly skilled educational writer. You supposed to write a concrete section of a lesson.
You are given a **Stakeholder Request**, **Lesson plan**, **Text written**, and **What to write**. Please don't forget to follow **Constraints**.
You are supposed to output only text that will be inserted into original html.
""")
    LESSON_ELEMENT_CONSTRAINTS = PromptTemplate.from_template("""
- You must use language of request of **Stakeholder Request**. If you use info from **Relevant info** you must translate it to language of **Stakeholder Request**
- You MUST use HTML instead of markdown
- You must highlight words to focus student attention on specific parts. Use for this <strong>, <em>, ...
- You can use <br> for separation
- Please cover all points of specific section
- If what to write contains some exercise than you must improve this exercise. Make exercise to be solvable from the knowledge from the text. In such case you must return only improved task, do not add any other text. Don't write in the task that it was improved. DON'T SOLVE EXERCISE.
- If what to write contains some paragraph that explains something you should improve and expand information in it
""")
    
    text_written = deepcopy(lesson_plan)
    def traverse_struct(struct):
        tag_name, struct = list(struct.items())[0]
        if 'children' in struct:
            for child in struct['children']:
                traverse_struct(child)
        if 'id' in struct and 'text' in struct and tag_name=='p':
            if text_files_manager:
                relevant_info_from_files = simple_search_info_for_lesson(model,struct['text'] , text_files_manager, user_id, doc_id)
            LESSON_ELEMENT_TEMPLATE = PromptTemplate.from_template(
                GENERAL_PIPELINE_TEMPLATE.invoke(
                    {
                        'context': LESSON_ELEMENT_CONTEXT.template + f'\nRelevant info:\n\n{relevant_info_from_files}\n\n',
                        'role': LESSON_ELEMENT_ROLE.template,
                        'constraints': LESSON_ELEMENT_CONSTRAINTS.template
                    }
                ).text
            )
            struct['text'] = model.invoke(
                LESSON_ELEMENT_TEMPLATE.format(
                    stakeholder_request = stakeholder_request,
                    lesson_plan = yaml.dump(lesson_plan, indent=2),
                    text_written = yaml.dump(text_written, indent=2),
                    what_to_write = struct['text']
                )
            ).content
            struct['text'] = extract_code_from_ai_response(struct['text'])
    traverse_struct(text_written)
    return text_written
   


if __name__ == '__main__':
   print('Full pipeline')
   from dotenv import load_dotenv
   load_dotenv()
   from langchain_google_genai import ChatGoogleGenerativeAI
   from chroma_utils import ChromaTextFilesManager
   from chromadb import Client
   import os
   from tqdm import tqdm
   from PyPDF2 import PdfReader

   chroma_client = Client()
   text_files_manager = ChromaTextFilesManager(chroma_client)
   pdf_path = os.path.join("data", "documents", "lbdl.pdf")
   doc_id = "lbdl"
   user_id = "user123"
   reader = PdfReader(pdf_path)
   text = ""
   for page in tqdm(reader.pages):
        text += page.extract_text() + "\n"
   text_files_manager.add_text(text=text, user_id=user_id, doc_id=doc_id)

   model = ChatGoogleGenerativeAI(model='models/gemini-2.0-flash', temperature=0)
   
   style_request = 'Стиль: Шрифт строгий и футуристичный, в светлой теме, строгий стиль\nСтруктура: Три параграфа с заголовками и секция с заданиями'
   lesson_theme = 'Урок про ML'

   convert_style = convert_style(model, style_request)
   html_template = generate_lesson_design(model, convert_style)

   relevant_info_from_files = simple_search_info_for_lesson(model, lesson_theme, text_files_manager, user_id, doc_id)

   lesson_plan = generate_lesson_plan(model, lesson_theme, html_template, relevant_info_from_files)
   html_content = fill_html_with_text(html_template, lesson_plan)
   print('Stage 1')

   improved_lesson_plan = improve_text(model, lesson_theme, lesson_plan, user_id, doc_id, text_files_manager)
   improved_html_content = fill_html_with_text(html_template, improved_lesson_plan)
   improved_html_content = extract_code_from_ai_response(improved_html_content)
   print('Stage 2')
   print(improved_html_content)

   with open('improved_html.html', 'w', encoding='utf-8') as f:
      f.write(improved_html_content)
