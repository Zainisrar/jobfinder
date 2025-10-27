from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from google import genai
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import json
import os
from bson import ObjectId
from werkzeug.utils import secure_filename
from typing import List
import logging
from datetime import datetime
from io import BytesIO
# New imports for document processing and chatbot
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from PIL import Image
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone
# PDF generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# ✅ Gemini client setup
def get_gemini_client():
    return genai.Client(api_key="AIzaSyAP3gfWeblgHNmc0GSCLID3h8HMf4cRyGU")

gemini_client = get_gemini_client()

# ✅ Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ✅ Environment variables and configuration
GEMINI_API_KEY = os.getenv("GENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DOMAIN_NAME = "chatbots"  # Static domain name

# Initialize Pinecone and embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index = pc.Index(DOMAIN_NAME)

# Debug: Print API key info
print(f"🔑 Loaded OpenAI API Key: {OPENAI_API_KEY[:20] + '...' + OPENAI_API_KEY[-4:] if OPENAI_API_KEY else 'NOT FOUND'}")
print(f"🔑 Loaded Gemini API Key: {GEMINI_API_KEY[:20] + '...' + GEMINI_API_KEY[-4:] if GEMINI_API_KEY else 'NOT FOUND'}")

# Set environment variables
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print(f"✅ Set OPENAI_API_KEY in environment: {OPENAI_API_KEY[:20]}...{OPENAI_API_KEY[-4:]}")

# ✅ Global conversation instance
conversation_instance = None
# ✅ Folder to store uploaded CVs locally (or replace with S3/Firebase later)
UPLOAD_FOLDER = "uploads"
REPORTS_FOLDER = "reports"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(REPORTS_FOLDER):
    os.makedirs(REPORTS_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORTS_FOLDER"] = REPORTS_FOLDER
# ✅ MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.error("MONGO_URI not found in environment variables")
    raise ValueError("MONGO_URI is required")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["personality_test_db"]
collection = db["MCQs"]
users_collection = db["users"]  # ✅ New collection for user auth
results_collection = db["results"]
jobs_collection = db["jobs"]   # ✅ New collection for jobs
user_data_collection = db["userData"]

# Verify API keys are loaded
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables")
if not GEMINI_API_KEY:
    logger.error("GENAI_API_KEY not found in environment variables")
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY not found in environment variables")
# ✅ Document processing and chatbot functions
def get_vector_store(text_chunks: List[str]):
    """Create vector store from text chunks"""
    try:
        # Ensure we're using the correct API key
        api_key = OPENAI_API_KEY  # Use the global variable instead of os.getenv
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        print(f"🔍 Using API key in get_vector_store: {api_key[:20]}...{api_key[-4:]}")

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise e

def create_conversation_chain(vectorstore, streaming=False, callback=None):
    """Create conversational retrieval chain with optional streaming"""
    try:
        # Configure callbacks properly
        callbacks = [callback] if callback else []

        llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        prompt_template = """You are a helpful AI assistant for the recommended courses. Use the provided context to answer recommended 3 courses with name and description accurately as possible.
If the answer is not in the context, recommended 3 courses with course code.

Context:
{context}

Chat History:
{chat_history}

User's Question:
{question}

Your Answer:"""

        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=prompt_template
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=False,
            verbose=False
        )

        return conversation_chain
    except Exception as e:
        logger.error(f"Error creating conversation chain: {str(e)}")
        raise e

def split_documents(pages: List[Document]) -> List[str]:
    """Split documents into chunks"""
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        split_docs = text_splitter.split_documents(pages)
        texts = [doc.page_content for doc in split_docs]
        return texts
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise e

def extract_text_from_pdf(file_path: str) -> List[str]:
    """Extract text from PDF file"""
    try:
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        split_docs = split_documents(pages)
        return split_docs
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise e

def process_file(file_path: str) -> List[str]:
    """Process different file types"""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            return extract_text_from_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise e

# ✅ Helper function to generate questions
def generate_questions(trait):
    prompt = f"""
    Generate 20 personality test questions in JSON format.
    Each question should measure trait '{trait}'.
    Follow this structure:
    {{
      "question_no": 1,
      "trait": "{trait}",
      "question": "I feel confident expressing my ideas in a group.",
      "options": [
        {{ "text": "Strongly Agree", "score": 5 }},
        {{ "text": "Agree", "score": 4 }},
        {{ "text": "Neutral", "score": 3 }},
        {{ "text": "Disagree", "score": 2 }},
        {{ "text": "Strongly Disagree", "score": 1 }}
      ]
    }}
    Return ONLY valid JSON array of 20 questions.
    Note: Add some negative questions also.
    """

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    output_text = response.candidates[0].content.parts[0].text.strip()

    if output_text.startswith("```"):
        output_text = output_text.strip("`")
        if output_text.lower().startswith("json"):
            output_text = output_text[4:].strip()

    return json.loads(output_text)

# ============================
# ✅ NEW API - USER SIGNUP
# ============================
@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")
        role = data.get("role")

        if not all([name, email, password, role]):
            return jsonify({"error": "All fields are required"}), 400

        # Check if user already exists
        if users_collection.find_one({"email": email}):
            return jsonify({"error": "User already exists"}), 409

        hashed_password = generate_password_hash(password)

        user = {
            "name": name,
            "email": email,
            "password": hashed_password,
            "role": role
        }

        users_collection.insert_one(user)
        return jsonify({"message": "User registered successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================
# ✅ NEW API - USER SIGNIN

@app.route("/signin", methods=["POST"])
def signin():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "User not found"}), 404

        if not check_password_hash(user["password"], password):
            return jsonify({"error": "Invalid credentials"}), 401

        return jsonify({
            "message": "Login successful",
            "user": {
                "id": str(user["_id"]),  # ✅ Convert ObjectId to string
                "name": user["name"],
                "email": user["email"],
                "role": user["role"]
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ API route
@app.route("/generate_test", methods=["POST"])
def generate_test():
    data = request.json
    user_id = data.get("user_id")
    trait = "Communication"  # For simplicity, fixed trait. Can be extended.

    if not user_id or not trait:
        return jsonify({"error": "user_id and trait are required"}), 400

    try:
        questions = generate_questions(trait)

        document = {
            "user_id": user_id,
            "trait": trait,
            "test_name": "Personality Test",
            "questions": questions
        }

        result = collection.insert_one(document)

        return jsonify({
            "message": "Test generated successfully",
            "mcqs_id": str(result.inserted_id),
            "trait": trait,
            "questions": questions
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ NEW GET API to fetch all MCQs by user_id
@app.route("/get_mcqs/<user_id>", methods=["GET"])
def get_mcqs(user_id):
    docs = list(collection.find({"user_id": user_id}, {"_id": 0}))  # hide _id for clean response
    if not docs:
        return jsonify({"error": "No MCQs found for this user"}), 404
    return jsonify(docs)

# ---------- Helper Functions ----------
def calculate_score(questions, user_answers):
    total_score = 0
    max_score = 0
    trait_scores = {}

    for q in questions:
        q_no = q["question_no"]
        trait = q["trait"]
        max_score += 5  # every question max = 5

        if str(q_no) in user_answers:  # answers may come as string keys
            selected = user_answers[str(q_no)]
            for option in q["options"]:
                if option["text"] == selected:
                    score = option["score"]
                    total_score += score

                    if trait not in trait_scores:
                        trait_scores[trait] = {"score": 0, "max": 0}
                    trait_scores[trait]["score"] += score
                    trait_scores[trait]["max"] += 5
                    break

    return total_score, max_score, trait_scores

def analyze_traits(trait_scores):
    analysis = {}
    for trait, data in trait_scores.items():
        percent = (data["score"] / data["max"]) * 100
        if percent >= 70:
            analysis[trait] = "Strength"
        elif percent <= 40:
            analysis[trait] = "Weakness"
        else:
            analysis[trait] = "Neutral"
    return analysis

# ---------- API Endpoints ----------

def convert_objectid(obj):
    if isinstance(obj, list):
        return [convert_objectid(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_objectid(v) for k, v in obj.items()}
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj

@app.route("/submit_answers", methods=["POST"])
def submit_answers():
    data = request.json
    user_id = data.get("user_id")
    mcq_id = data.get("mcq_id")  # ✅ From frontend
    user_answers = data.get("answers")

    if not user_id or not mcq_id or not user_answers:
        return jsonify({"error": "user_id, mcq_id and answers are required"}), 400

    # Fetch questions using mcq_id
    doc = collection.find_one({"_id": ObjectId(mcq_id)})
    if not doc:
        return jsonify({"error": "No MCQ document found for this mcq_id"}), 404

    questions = doc["questions"]

    # Calculate scores
    total_score, max_score, trait_scores = calculate_score(questions, user_answers)
    analysis = analyze_traits(trait_scores)

    # Prepare result document
    result_doc = {
        "user_id": user_id,
        "mcq_id": mcq_id,
        "total_score": total_score,
        "max_score": max_score,
        "percentage": round((total_score / max_score) * 100, 2),
        "trait_scores": trait_scores,
        "analysis": analysis
    }

    # Store result in DB
    insert_result = results_collection.insert_one(result_doc)

        # ✅ Convert inserted_id to string for JSON
    response_doc = convert_objectid(result_doc)
    response_doc["result_id"] = str(insert_result.inserted_id)

    # ✅ Filter only useful fields for report
    filtered_response = {
        "user_id": response_doc.get("user_id"),
        "mcq_id": response_doc.get("mcq_id"),
        "result_id": response_doc.get("result_id"),
        "total_score": response_doc.get("total_score"),
        "max_score": response_doc.get("max_score"),
        "percentage": response_doc.get("percentage"),
        "analysis": response_doc.get("analysis")
    }

    return jsonify({
        "message": "Results submitted and saved successfully",
        "data": filtered_response
    }), 201

@app.route("/get_result_by_id", methods=["GET"])
def get_result_by_id():
    try:
        result_id = request.args.get("result_id")

        if not result_id:
            return jsonify({"error": "result_id is required"}), 400

        # ✅ Fetch result from DB using ObjectId
        result = results_collection.find_one({"_id": ObjectId(result_id)})

        if not result:
            return jsonify({"error": "No result found for this result_id"}), 404

        # ✅ Convert ObjectId to string
        result["_id"] = str(result["_id"])
        result["result_id"] = result["_id"]

        # ✅ Filter only useful fields
        filtered_result = {
            "user_id": result.get("user_id"),
            "mcq_id": result.get("mcq_id"),
            "result_id": result.get("result_id"),
            "total_score": result.get("total_score"),
            "max_score": result.get("max_score"),
            "percentage": result.get("percentage"),
            "analysis": result.get("analysis")
        }

        return jsonify({
            "message": "Result fetched successfully",
            "data": filtered_result
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------
# job api
# -----------------------------------
# ✅ Create Job API
@app.route("/jobs", methods=["POST"])
def create_job():
    try:
        data = request.get_json()

        job = {
            "title": data.get("title"),
            "company": data.get("company"),
            "location": data.get("location"),
            "type": data.get("type"),
            "salary": data.get("salary"),
            "description": data.get("description"),
            "skills": data.get("skills", [])
        }

        result = jobs_collection.insert_one(job)
        job["_id"] = str(result.inserted_id)  # ✅ convert ObjectId to string

        return jsonify({"message": "Job created successfully!", "job": job}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from bson import ObjectId

def serialize_doc(doc):
    """Convert MongoDB document into JSON-serializable dict"""
    if doc is None:
        return None
    
    # Create a copy to avoid modifying the original
    serialized = {}
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            serialized[key] = str(value)
        elif isinstance(value, dict):
            serialized[key] = serialize_doc(value)
        elif isinstance(value, list):
            serialized[key] = [serialize_doc(item) if isinstance(item, dict) else str(item) if isinstance(item, ObjectId) else item for item in value]
        else:
            serialized[key] = value
    return serialized

@app.route("/jobs", methods=["GET"])
def get_jobs():
    try:
        jobs = [serialize_doc(job) for job in jobs_collection.find()]
        return jsonify(jobs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jobs/<job_id>", methods=["PUT"])
def update_job(job_id):
    try:
        data = request.get_json()

        update_fields = {}
        if data.get("title"):
            update_fields["title"] = data.get("title")
        if data.get("company"):
            update_fields["company"] = data.get("company")
        if data.get("location"):
            update_fields["location"] = data.get("location")
        if data.get("type"):
            update_fields["type"] = data.get("type")
        if data.get("salary"):
            update_fields["salary"] = data.get("salary")
        if data.get("description"):
            update_fields["description"] = data.get("description")
        if "skills" in data:
            update_fields["skills"] = data.get("skills", [])

        if not update_fields:
            return jsonify({"error": "No fields to update"}), 400

        result = jobs_collection.update_one(
            {"_id": ObjectId(job_id)}, {"$set": update_fields}
        )

        if result.matched_count == 0:
            return jsonify({"error": "Job not found"}), 404

        updated_job = jobs_collection.find_one({"_id": ObjectId(job_id)})
        return jsonify({"message": "Job updated successfully!", "job": serialize_doc(updated_job)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jobs/<job_id>", methods=["DELETE"])
def delete_job(job_id):
    try:
        result = jobs_collection.delete_one({"_id": ObjectId(job_id)})

        if result.deleted_count == 0:
            return jsonify({"error": "Job not found"}), 404

        return jsonify({"message": "Job deleted successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------
# data Store
# -----------------------------------------------

def extract_text_from_cv(cv_path):
    """Extract text content from PDF CV"""
    try:
        pdf_reader = PdfReader(cv_path)
        cv_text = ""
        for page in pdf_reader.pages:
            cv_text += page.extract_text() + "\n"
        return cv_text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from CV: {str(e)}")
        return ""

def llm_recommendation(job_data, cv_text):
    """Generate AI recommendation based on job requirements and CV content"""
    import time
    
    prompt = f"""
    You are an expert HR recruiter. Analyze the candidate's CV against the job requirements.

    Job Details:
    - Title: {job_data.get('title', 'N/A')}
    - Company: {job_data.get('company', 'N/A')}
    - Required Skills: {', '.join(job_data.get('skills', []))}
    - Description: {job_data.get('description', 'N/A')}

    Candidate's CV:
    {cv_text[:2000]}

    Provide ONLY one of these responses:
    - "Recommended" - if the candidate matches 60% or more of the requirements
    - "Not Recommended" - if the candidate matches less than 60% of the requirements

    Return only the recommendation status, nothing else.
    """

    # Try Gemini first with retry logic
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            recommendation = response.candidates[0].content.parts[0].text.strip()
            return recommendation
        except Exception as gemini_error:
            error_msg = str(gemini_error)
            logger.warning(f"Gemini attempt {attempt + 1} failed: {error_msg}")
            
            if "503" in error_msg or "overloaded" in error_msg.lower():
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait 2 seconds before retry
                    continue
            break
    
    # Fallback to OpenAI if Gemini fails
    try:
        logger.info("Falling back to OpenAI for recommendation generation")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as openai_error:
        logger.error(f"OpenAI fallback also failed: {str(openai_error)}")
    
    # Final fallback: Basic rule-based recommendation
    try:
        required_skills = set(skill.lower() for skill in job_data.get('skills', []))
        cv_lower = cv_text.lower()
        
        matched_skills = [skill for skill in required_skills if skill in cv_lower]
        match_percentage = int((len(matched_skills) / len(required_skills) * 100)) if required_skills else 0
        
        return "Recommended" if match_percentage >= 60 else "Not Recommended"
        
    except Exception as e:
        logger.error(f"All recommendation methods failed: {str(e)}")
        return "Not Recommended"

@app.route("/get-all-users", methods=["GET"])
def get_all_users():
    try:
        users = [serialize_doc(user) for user in user_data_collection.find()]
        return jsonify({"message": "Users fetched successfully!", "users": users}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get-results-by-user/<user_id>", methods=["GET"])
def get_results_by_user(user_id):
    try:
        results = list(results_collection.find({"user_id": user_id}))
        
        if not results:
            return jsonify({"message": "No results found for this user", "results": []}), 200
        
        serialized_results = [serialize_doc(result) for result in results]
        return jsonify({"message": "Results fetched successfully!", "results": serialized_results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/submit-user-data", methods=["POST"])
def submit_user_data():
    try:
        # ✅ Parse form data
        user_id = request.form.get("user_id")
        name = request.form.get("name")
        email = request.form.get("email")
        phoneNo = request.form.get("phoneNo")
        workexperience = request.form.get("workexperience")
        skills = request.form.get("skills")
        jobid = request.form.get("jobid")
        educationalbackground = request.form.get("educationalbackground")

        # ✅ Handle CV upload
        cv_file = request.files.get("CVupload")
        cv_path = None
        cv_text = ""
        
        if cv_file:
            filename = secure_filename(cv_file.filename)
            cv_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            cv_file.save(cv_path)  # save file locally
            
            # Extract text from PDF
            cv_text = extract_text_from_cv(cv_path)

        # ✅ Get job data from job collection
        job_data = None
        recommendation = "No recommendation available"
        
        if jobid:
            try:
                job_data = jobs_collection.find_one({"_id": ObjectId(jobid)})
                
                if job_data and cv_text:
                    # Generate AI recommendation
                    recommendation = llm_recommendation(job_data, cv_text)
                elif not cv_text:
                    recommendation = "CV not provided - unable to generate recommendation"
                else:
                    recommendation = "Job not found - unable to generate recommendation"
                    
            except Exception as job_error:
                logger.error(f"Error fetching job data: {str(job_error)}")
                recommendation = f"Error processing job data: {str(job_error)}"

        # ✅ Create user document
        user_doc = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "phoneNo": phoneNo,
            "ai_recommendation": recommendation,
            "workexperience": workexperience,
            "skills": skills,
            "jobid": jobid,
            "educationalbackground": educationalbackground,
            "CVupload": cv_path  # saved path
        }

        # ✅ Save to MongoDB
        result = user_data_collection.insert_one(user_doc)

        return jsonify({
            "message": "User data submitted successfully!",
            "id": str(result.inserted_id),
            "cv_path": cv_path,
            "ai_recommendation": recommendation
        }), 201

    except Exception as e:
        logger.error(f"Error in submit_user_data: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ✅ Chatbot API endpoint
@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Ask a question about the document"""
    global conversation_instance
    try:
        data = request.get_json()
        question = data.get("question")

        if not conversation_instance:
            return jsonify({"error": "No document has been processed. Please upload a document first."}), 400

        if not question or not question.strip():
            return jsonify({"error": "Question cannot be empty"}), 400

        logger.info(f"Processing question: {question}")

        # Get response from conversation chain
        response = conversation_instance({'question': question})
        answer = response.get('answer', 'No answer available')

        logger.info("Question processed successfully")

        return jsonify({
            "answer": answer,
            "success": True,
            "message": "Question processed successfully"
        }), 200

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# ✅ Document upload and processing endpoint
@app.route("/upload-document", methods=["POST"])
def upload_document():
    """Upload and process document for chatbot"""
    global conversation_instance
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Process the document
        text_chunks = process_file(file_path)

        # Create vector store
        vectorstore = get_vector_store(text_chunks)

        # Create conversation chain
        conversation_instance = create_conversation_chain(vectorstore)

        logger.info(f"Document processed successfully: {filename}")

        return jsonify({
            "message": "Document uploaded and processed successfully",
            "filename": filename,
            "success": True
        }), 200

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return jsonify({"error": f"Error processing document: {str(e)}"}), 500


# ✅ Course Recommendation API
@app.route('/recommend-courses', methods=['POST'])
def recommend_courses():
    """
    Recommend top 3 courses based on user score and interested field

    Expected JSON input:
    {
        "score": 85,
        "interested_field": "Computer Science"
    }

    Returns:
    {
        "status": "success",
        "data": {
            "courses": [
                {
                    "name": "Course Name",
                    "code": "CS101",
                    "description": "Course description"
                }
            ],
            "user_input": {
                "score": 85,
                "interested_field": "Computer Science"
            }
        }
    }
    """
    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400

        score = data.get('score')
        interested_field = data.get('interested_field')

        # Validate input
        if score is None or not interested_field:
            return jsonify({
                "status": "error",
                "message": "Missing required fields: 'score' and 'interested_field'"
            }), 400

        # Validate score range
        try:
            score = float(score)
            if score < 0 or score > 100:
                return jsonify({
                    "status": "error",
                    "message": "Score must be between 0 and 100"
                }), 400
        except ValueError:
            return jsonify({
                "status": "error",
                "message": "Score must be a valid number"
            }), 400

        # Create query
        query = f"Recommend top courses for a student with score {score} interested in {interested_field}. Provide course names and course codes."

        # Create query embedding
        query_vector = embeddings.embed_query(query)

        # Query Pinecone
        query_response = index.query(
            vector=query_vector,
            top_k=10,  # Get more results to filter through
            include_metadata=True
        )

        matches = query_response.get("matches", [])

        if not matches:
            return jsonify({
                "status": "success",
                "message": "No courses found",
                "data": {
                    "courses": [],
                    "user_input": {
                        "score": score,
                        "interested_field": interested_field
                    }
                }
            }), 200

        # Create documents from matches
        documents = []
        for match in matches:
            text = match["metadata"].get("text", "")
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata=match["metadata"]
                ))

        if not documents:
            return jsonify({
                "status": "error",
                "message": "Found matches but no text content available"
            }), 500

        # Create prompt for LLM
        prompt_template = """
You are a course recommendation expert. Based on the student's score and interested field, recommend the top 3 most suitable courses from the provided context.

Student Information:
- Score: {score}
- Interested Field: {interested_field}

Context from course catalog:
{context}

Instructions:
1. Analyze the courses available in the context
2. Select the top 3 courses that best match the student's score and interested field
3. Return ONLY a JSON array with exactly 3 courses
4. Each course must include: name, code, and description
5. If course codes are not explicitly mentioned, infer them from course names (e.g., "Introduction to Computer Science" -> "CS101")

Return ONLY valid JSON in this exact format with no additional text:
[
    {{"name": "Course Name 1", "code": "COURSE101", "description": "Brief description"}},
    {{"name": "Course Name 2", "code": "COURSE102", "description": "Brief description"}},
    {{"name": "Course Name 3", "code": "COURSE103", "description": "Brief description"}}
]
"""

        prompt = PromptTemplate(
            input_variables=["context", "score", "interested_field"],
            template=prompt_template,
        )

        # Get LLM response
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

        response = qa_chain.invoke({
            "context": documents,
            "score": score,
            "interested_field": interested_field
        })

        # Parse the response
        import json
        try:
            # Clean the response (remove markdown code blocks if present)
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()

            courses = json.loads(clean_response)

            # Ensure we have exactly 3 courses
            if len(courses) > 3:
                courses = courses[:3]

            return jsonify({
                "status": "success",
                "data": {
                    "courses": courses,
                    "user_input": {
                        "score": score,
                        "interested_field": interested_field
                    }
                }
            }), 200

        except json.JSONDecodeError:
            # Fallback: return raw response
            return jsonify({
                "status": "success",
                "data": {
                    "raw_response": response,
                    "user_input": {
                        "score": score,
                        "interested_field": interested_field
                    }
                },
                "message": "Unable to parse structured response"
            }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ✅ PDF Generation Function
def create_score_chart(score_summary):
    """Create a score visualization chart"""
    try:
        # Set matplotlib to use non-interactive backend
        plt.switch_backend('Agg')

        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('white')

        # Chart 1: Score Progress Bar
        percentage = score_summary.get('percentage', 0)
        total_score = score_summary.get('total_score', 0)
        max_score = score_summary.get('max_score', 100)

        # Progress bar chart
        bar_color = '#4CAF50' if percentage >= 70 else '#FF9800' if percentage >= 50 else '#F44336'
        bars = ax1.barh(['Score'], [percentage], color=bar_color, height=0.6)
        ax1.set_xlim(0, 100)
        ax1.set_xlabel('Percentage (%)', fontsize=10)
        ax1.set_title(f'Overall Score: {total_score}/{max_score} ({percentage}%)', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Add percentage text on bar
        if percentage > 10:  # Only show text if bar is wide enough
            ax1.text(percentage/2, 0, f'{percentage}%', ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        else:
            ax1.text(percentage + 5, 0, f'{percentage}%', ha='left', va='center', fontweight='bold', color='black', fontsize=10)

        # Chart 2: Trait Analysis Pie Chart
        analysis = score_summary.get('analysis', {})
        if analysis:
            trait_counts = {'Strength': 0, 'Weakness': 0, 'Neutral': 0}
            for trait, result in analysis.items():
                if result in trait_counts:
                    trait_counts[result] += 1

            # Filter out zero values
            filtered_counts = {k: v for k, v in trait_counts.items() if v > 0}

            if filtered_counts:
                colors_map = {'Strength': '#4CAF50', 'Weakness': '#F44336', 'Neutral': '#FF9800'}
                pie_colors = [colors_map[label] for label in filtered_counts.keys()]

                wedges, texts, autotexts = ax2.pie(filtered_counts.values(),
                                                 labels=filtered_counts.keys(),
                                                 colors=pie_colors,
                                                 autopct='%1.0f',
                                                 startangle=90,
                                                 textprops={'fontsize': 9})
                ax2.set_title('Trait Distribution', fontsize=12, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No Analysis\nData Available', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=10)
                ax2.set_title('Trait Distribution', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Analysis\nData Available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=10)
            ax2.set_title('Trait Distribution', fontsize=12, fontweight='bold')

        plt.tight_layout()

        # Save to BytesIO
        chart_buffer = BytesIO()
        plt.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        chart_buffer.seek(0)
        plt.close(fig)  # Close the figure to free memory

        return chart_buffer

    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        # Return None if chart creation fails
        return None


def create_pdf_report(report_data):
    """Generate a professional PDF report"""
    try:
        # Create a BytesIO buffer to store PDF
        buffer = BytesIO()

        # Create document with professional margins
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                              topMargin=60, bottomMargin=60)

        # Get styles
        styles = getSampleStyleSheet()

        # Professional custom styles
        title_style = ParagraphStyle(
            'ProfessionalTitle',
            parent=styles['Heading1'],
            fontSize=22,
            spaceAfter=25,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1F4E79'),
            fontName='Helvetica-Bold'
        )

        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#5A5A5A'),
            fontName='Helvetica'
        )

        section_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=18,
            textColor=colors.HexColor('#1F4E79'),
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor('#E5E5E5'),
            borderPadding=(0, 0, 3, 0),
            keepWithNext=True
        )

        content_style = ParagraphStyle(
            'Content',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor('#2C2C2C'),
            fontName='Helvetica',
            leading=16,
            firstLineIndent=0
        )

        highlight_style = ParagraphStyle(
            'Highlight',
            parent=styles['Normal'],
            fontSize=13,
            spaceAfter=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1F4E79'),
            fontName='Helvetica-Bold',
            backColor=colors.HexColor('#F7F9FC'),
            borderWidth=1,
            borderColor=colors.HexColor('#D1D9E6'),
            borderPadding=12
        )

        info_style = ParagraphStyle(
            'InfoStyle',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_LEFT,
            textColor=colors.HexColor('#4A4A4A'),
            fontName='Helvetica'
        )

        # Story list to hold all elements
        story = []

        # Professional Title Section
        story.append(Paragraph("PERSONALITY ASSESSMENT REPORT", title_style))
        story.append(Paragraph(f"Prepared for {report_data.get('user_name', 'Candidate')} | {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
        story.append(Spacer(1, 25))

        # Executive Summary
        score_summary = report_data.get('score_summary', {})
        percentage = score_summary.get('percentage', 0)
        grade = score_summary.get('grade', 'N/A')

        score_text = f"Overall Assessment Score: {score_summary.get('total_score', 0)}/{score_summary.get('max_score', 100)} ({percentage}%) | Performance Grade: {grade}"
        story.append(Paragraph(score_text, highlight_style))
        story.append(Spacer(1, 20))

        # Report Information
        story.append(Paragraph("Report Information", section_style))
        story.append(Paragraph(f"Report ID: {report_data.get('report_id', 'N/A')}", info_style))
        story.append(Paragraph(f"Assessment Type: {report_data.get('report_type', 'Personality Assessment').replace('_', ' ').title()}", info_style))
        story.append(Paragraph(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", info_style))
        story.append(Spacer(1, 15))

        # Performance Analysis Section
        story.append(Paragraph("Performance Analysis", section_style))

        # Create and add chart
        chart_buffer = None
        temp_chart_path = None
        try:
            chart_buffer = create_score_chart(score_summary)
            if chart_buffer:
                temp_chart_path = os.path.join(app.config.get("REPORTS_FOLDER", "reports"),
                                             f"temp_chart_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")

                with open(temp_chart_path, 'wb') as f:
                    f.write(chart_buffer.getvalue())

                # Center the chart
                chart_img = Image(temp_chart_path, width=5.5*inch, height=2.2*inch)
                story.append(chart_img)
                story.append(Spacer(1, 18))
        except Exception as chart_error:
            logger.error(f"Chart creation failed: {str(chart_error)}")
            story.append(Paragraph("Performance visualization data is being processed and will be available in future reports.", content_style))
            story.append(Spacer(1, 18))

        # Assessment Results
        analysis = score_summary.get('analysis', {})
        if analysis:
            story.append(Paragraph("Assessment Results", section_style))

            for trait, result in analysis.items():
                if result == 'Strength':
                    status = "Above Average"
                    color = colors.HexColor('#2E7D32')
                elif result == 'Weakness':
                    status = "Development Area"
                    color = colors.HexColor('#C62828')
                else:
                    status = "Average Range"
                    color = colors.HexColor('#F57C00')

                trait_text = f"<font color='{color.hexval()}'><b>{trait}:</b> {status}</font>"
                story.append(Paragraph(trait_text, content_style))

            story.append(Spacer(1, 18))

        # Performance Classification
        story.append(Paragraph("Performance Classification", section_style))
        if percentage >= 80:
            performance_text = "<b>Exceptional Performance</b><br/><br/>The assessment results indicate exceptional capabilities and comprehensive understanding of the evaluated competencies. This performance level demonstrates mastery and readiness for advanced responsibilities."
        elif percentage >= 60:
            performance_text = "<b>Proficient Performance</b><br/><br/>The assessment results show proficient capabilities with solid foundational understanding. This performance level indicates competence with opportunities for continued professional development."
        else:
            performance_text = "<b>Developing Performance</b><br/><br/>The assessment results suggest areas for focused development and skill enhancement. This performance level provides a clear foundation for targeted improvement initiatives."

        story.append(Paragraph(performance_text, content_style))
        story.append(Spacer(1, 18))

        # Professional Insights
        story.append(Paragraph("Professional Development Insights", section_style))

        detailed_report = report_data.get('detailed_report', '')
        if detailed_report:
            # Clean and format the report professionally
            paragraphs = detailed_report.split('\n')
            for para in paragraphs[:6]:
                if para.strip():
                    clean_para = para.strip().replace('*', '').replace('#', '').replace('**', '').replace('•', '')
                    if clean_para and len(clean_para) > 15:
                        # Ensure professional language
                        if clean_para.lower().startswith(('you ', 'your ')):
                            clean_para = clean_para[0].upper() + clean_para[1:]
                        story.append(Paragraph(clean_para, content_style))
        else:
            story.append(Paragraph("Comprehensive professional insights are generated based on individual assessment responses and performance patterns. These insights provide targeted recommendations for career development and skill enhancement.", content_style))

        story.append(Spacer(1, 18))

        # Recommendations
        story.append(Paragraph("Professional Recommendations", section_style))

        recommendations = [
            "Conduct a comprehensive review of identified strengths to leverage existing capabilities effectively in current and future roles.",
            "Develop a structured professional development plan addressing specific growth areas identified through this assessment.",
            "Engage with mentors, supervisors, or professional coaches to gain additional perspectives on performance and development opportunities.",
            "Schedule periodic reassessment to monitor progress and adjust development strategies as professional goals evolve."
        ]

        for i, recommendation in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {recommendation}", content_style))

        story.append(Spacer(1, 25))

        # Professional Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#888888'),
            fontName='Helvetica',
            borderWidth=1,
            borderColor=colors.HexColor('#E5E5E5'),
            borderPadding=(8, 0, 0, 0)
        )

        story.append(Paragraph(f"Confidential Assessment Report | Document ID: {report_data.get('report_id', 'N/A')} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}", footer_style))

        # Build PDF
        doc.build(story)

        # Get the value of the BytesIO buffer
        pdf_data = buffer.getvalue()
        buffer.close()

        # Clean up resources
        if chart_buffer:
            chart_buffer.close()

        # Clean up temporary chart file
        if temp_chart_path and os.path.exists(temp_chart_path):
            try:
                os.remove(temp_chart_path)
            except Exception as cleanup_error:
                logger.warning(f"Could not remove temporary chart file: {cleanup_error}")

        return pdf_data

    except Exception as e:
        logger.error(f"Error creating PDF report: {str(e)}")
        raise e


# ✅ Generate Report API - Simplified with Result ID Only
@app.route("/generate-report", methods=["POST"])
def generate_report():
    """Generate a PDF report using only result_id - fetches all data from database"""
    try:
        data = request.get_json()

        # Get result_id from request
        result_id = data.get("result_id")

        if not result_id:
            return jsonify({"error": "result_id is required"}), 400

        # Fetch result data from database
        result = results_collection.find_one({"_id": ObjectId(result_id)})
        if not result:
            return jsonify({"error": "Result not found"}), 404

        # Extract data from database result
        analysis = result.get("analysis", {})
        max_score = result.get("max_score", 100)
        mcq_id = result.get("mcq_id")
        percentage = result.get("percentage", 0)
        total_score = result.get("total_score", 0)
        user_id = result.get("user_id")

        # Fetch user details
        user_details = users_collection.find_one({"_id": ObjectId(user_id)}) if user_id else None

        # Generate comprehensive report using Gemini AI
        report_prompt = f"""
        Generate a professional personality assessment report based on the following data:

        User Score: {total_score}/{max_score} ({percentage}%)
        Analysis Results: {analysis}

        Please provide a comprehensive report with:
        1. Executive Summary (2-3 sentences)
        2. Strengths and Development Areas
        3. Professional Recommendations
        4. Career Development Suggestions
        5. Action Steps for Growth

        Keep the report professional and actionable. Total report should be around 400-500 words.
        Format without markdown - use clear paragraphs and professional language.
        """

        # Generate report using Gemini
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=report_prompt
        )

        generated_report = response.candidates[0].content.parts[0].text.strip()

        # Prepare report data for PDF generation
        report_data = {
            "report_id": f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "result_id": result_id,
            "user_id": user_id,
            "mcq_id": mcq_id,
            "report_type": "personality_assessment",
            "generated_at": datetime.now().isoformat(),
            "score_summary": {
                "total_score": total_score,
                "max_score": max_score,
                "percentage": percentage,
                "grade": "A" if percentage >= 90 else "B" if percentage >= 80 else "C" if percentage >= 70 else "D" if percentage >= 60 else "F",
                "analysis": analysis
            },
            "detailed_report": generated_report,
            "user_name": user_details.get("name") if user_details else "Unknown User"
        }

        # Generate PDF directly
        pdf_data = create_pdf_report(report_data)

        # Create filename
        filename = f"personality_report_{user_details.get('name', 'user').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Create a BytesIO object to return the PDF
        pdf_buffer = BytesIO(pdf_data)
        pdf_buffer.seek(0)

        # Return the PDF file directly
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )

    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return jsonify({"error": f"Failed to generate PDF report: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500





# ✅ ASGI wrapper for compatibility with uvicorn
from asgiref.wsgi import WsgiToAsgi
asgi_app = WsgiToAsgi(app)

# ✅ Run Flask app
if __name__ == "__main__":
    print("🚀 Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)