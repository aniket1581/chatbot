import os
import requests
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain.schema import Document
import ollama

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# JSON file path
JSON_FILE_PATH = "clickup_data.json"

# ClickUp API key
CLICKUP_API_KEY = os.getenv("CLICKUP_API_KEY")

class QueryRequest(BaseModel):
    query: str

def _make_request(endpoint: str) -> dict:
    """Make a GET request to the ClickUp API"""
    url = f"https://api.clickup.com/api/v2/{endpoint}"
    try:
        logger.info(f"Requesting: {url}")
        response = requests.get(url, headers={
            "Authorization": CLICKUP_API_KEY,
            'Content-Type': 'application/json'
        })
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ClickUp API error: {str(e)}")

def save_to_json(documents: List[dict]):
    """Save documents to JSON file"""
    try:
        with open(JSON_FILE_PATH, 'w') as f:
            json.dump({"tasks": documents}, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving to JSON: {str(e)}")
        raise

def load_from_json() -> List[dict]:
    """Load documents from JSON file"""
    try:
        if not os.path.exists(JSON_FILE_PATH):
            return []
        with open(JSON_FILE_PATH, 'r') as f:
            data = json.load(f)
            return data.get("tasks", [])
    except Exception as e:
        logger.error(f"Error loading from JSON: {str(e)}")
        return []

def fetch_clickup_tasks() -> List[dict]:
    documents = []
    try:
        # Fetch spaces
        spaces = _make_request(f"team/3342101/space")
        for space in spaces.get("spaces", []):
            space_id = space["id"]
            space_name = space["name"]
            
            # Fetch folders in space
            folders = _make_request(f"space/{space_id}/folder")
            for folder in folders.get("folders", []):
                folder_id = folder["id"]
                folder_name = folder["name"]
                
                # Fetch lists in folder
                lists = _make_request(f"folder/{folder_id}/list")
                for list_obj in lists.get("lists", []):
                    list_id = list_obj["id"]
                    list_name = list_obj["name"]
                    
                    # Fetch tasks in list
                    tasks = _make_request(f"list/{list_id}/task")
                    for task in tasks.get("tasks", []):
                        task_data = {
                            "title": task.get('name', ''),
                            "description": task.get('description', ''),
                            "status": task.get('status', {}).get('status', 'Unknown'),
                            "priority": task.get('priority', {}).get('priority', 'Not set') if isinstance(task.get('priority', {}), dict) else 'Not set',
                            "due_date": task.get('due_date', 'None'),
                            "created_date": task.get('date_created', 'None'),
                            "assigned_to": task.get('assignees', []),
                            "task_id": task['id'],
                            "list_name": list_name,
                            "folder_name": folder_name,
                            "space_name": space_name,
                            "url": task.get('url', '')
                        }
                        documents.append(task_data)
    except Exception as e:
        logger.error(f"Error fetching ClickUp data: {str(e)}")
        raise
    return documents

@app.get("/ingest")
def ingest():
    try:
        tasks = fetch_clickup_tasks()
        save_to_json(tasks)
        return {"message": "Data ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_knowledge(req: QueryRequest):
    tasks = load_from_json()
    # Simple text matching for now - can be enhanced with more sophisticated search
    relevant_texts = []
    query = req.query.lower()
    
    for task in tasks:
        # Get title and description, defaulting to empty string if None
        title = task.get("title", "") or ""
        description = task.get("description", "") or ""
        
        # Basic search in title and description
        if query in title.lower() or query in description.lower():
            task_text = f"""
            Title: {task["title"] or "No title"}
            Description: {task["description"] or "No description"}
            Status: {task["status"] or "Unknown"}
            Priority: {task["priority"] or "Not set"}
            Due Date: {task["due_date"] or "None"}
            Created Date: {task["created_date"] or "None"}
            Assigned To: {task["assigned_to"] or "None"}
            """
            relevant_texts.append(task_text)

    if not relevant_texts:
        return {"response": "No matching tasks found."}

    formatted_tasks = "\n\n".join(relevant_texts)
    response = ollama.chat(
        model="deepseek-r1:14b", 
        messages=[
            {"role": "system", "content": "You are an AI assistant that retrieves tasks assigned to a user from ClickUp data."},
            {"role": "user", "content": f"\n".join(formatted_tasks)}
        ])
    return {"response": response}
