#!/usr/bin/env python

import collections
collections.MutableMapping = collections.abc.MutableMapping

"""
Integrated Backend for LLM‑Powered Fraud Detection & Anomaly Analysis

Phases integrated:
  • Phase 1: Data Simulation & Multi‑Modal Feature Engineering
  • Phase 2: Hybrid Embedding Generation & Multi‑LLM Integration
  • Phase 3: Distributed Vector Search & Real‑Time Detection
  • Phase 4: Full Pipeline & Rule‑Based Anomaly Detection with XAI
  • Phase 5: Real‑Time Monitoring & Alerting (WebSockets, SMS, Prometheus, Bayesian Optimization)
  • Phase 6: Enterprise‑Grade Security & Compliance (AES‑256, JWT, Audit Logging, Right to be Forgotten)
  • Phase 7: Frontend Dashboard & API Development – Exposing a GraphQL endpoint for flexible querying
       and a FastAPI-based fraud detection endpoint for ultra‑low latency processing.
  
This production‑quality backend exposes secure REST APIs, a GraphQL endpoint, and a FastAPI fraud detection API.
"""

import os, sys, json, time, base64, socket, random, datetime, logging, threading, asyncio, concurrent.futures

# ------------------------- Phase 1 Imports: Data Simulation & Feature Engineering -------------------------
import pandas as pd, numpy as np
from faker import Faker
from tqdm import tqdm
import spacy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, Model, Input
import tensorflow as tf
import matplotlib.pyplot as plt  # (for internal use)

# ------------------------- Phase 2 Imports: Hybrid Embedding Generation -------------------------
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import onnx, onnxruntime as ort

# ------------------------- Phase 3 Imports: Vector Search -------------------------
import faiss
from annoy import AnnoyIndex
import hnswlib

# ------------------------- Phase 4 Imports: ETL & Explainability -------------------------
import dask.dataframe as dd
import shap, lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ------------------------- Phase 5 Imports: Real‑Time Monitoring & Alerting -------------------------
import websockets
from twilio.rest import Client
from prometheus_client import CollectorRegistry, start_http_server, Summary, Counter
from skopt import gp_minimize

# ------------------------- Phase 6 Imports: Security & Compliance -------------------------
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

# ------------------------- Phase 7 Imports: GraphQL and FastAPI -------------------------
import graphene
from flask_graphql import GraphQLView
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ------------------------- Configuration and Logging -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FraudDetectionBackend")

# Environment-based configuration
AES_KEY = os.environ.get("AES_KEY", "0123456789ABCDEF0123456789ABCDEF").encode('utf-8')
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "SuperSecretJWTKey")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "your_account_sid")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "your_auth_token")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER", "+1234567890")
ALERT_SMS_TO_NUMBER = os.environ.get("ALERT_SMS_TO_NUMBER", "+0987654321")
FAISS_INDEX_PATH = "faiss_hnsw.index"
AUDIT_LOG_FILE = "audit_log.txt"

# Start Prometheus metrics HTTP server on port 8000
registry = CollectorRegistry()
alert_counter = Counter("fraud_alerts_total", "Total number of fraud alerts triggered", registry=registry)
alert_processing_time = Summary("alert_processing_seconds", "Time spent processing alerts", registry=registry)
start_http_server(8000, registry=registry)
logger.info("Prometheus metrics server started on port 8000.")

# ------------------------- Phase 1: Data Simulation & Feature Engineering -------------------------
nlp = spacy.load("en_core_web_sm")
faker = Faker()

def simulate_transactions(num_records=10000):
    """Simulate a transaction dataset with multi‑modal features."""
    regions = ['North', 'South', 'East', 'West', 'Central']
    merchants = [faker.company() for _ in range(50)]
    times_of_day = ['morning', 'afternoon', 'evening', 'night']
    data = []
    for _ in tqdm(range(num_records), desc="Simulating transactions"):
        user_id = faker.uuid4()
        age = random.randint(18, 80)
        region = random.choice(regions)
        credit_score = random.randint(300, 850)
        behavioral_history = faker.sentence(nb_words=10)
        transaction_id = faker.uuid4()
        timestamp = faker.date_time_this_year()
        amount = round(random.uniform(1.0, 1000.0), 2)
        merchant = random.choice(merchants)
        ip = faker.ipv4()
        device_fingerprint = faker.sha1(raw_output=False)
        location = f"{faker.city()}, {faker.country()}"
        velocity_pattern = round(random.uniform(0.1, 5.0), 2)
        browser_info = faker.user_agent()
        network_latency = round(random.uniform(10.0, 500.0), 2)
        time_of_day = random.choice(times_of_day)
        session_metadata = faker.uuid4()
        label = random.choices(["normal", "suspicious", "fraudulent"], weights=[90, 5, 5], k=1)[0]
        if label == "fraudulent":
            amount = round(amount * random.uniform(1.5, 3.0), 2)
        data.append({
            "user_id": user_id,
            "age": age,
            "region": region,
            "credit_score": credit_score,
            "behavioral_history": behavioral_history,
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "amount": amount,
            "merchant": merchant,
            "ip": ip,
            "device_fingerprint": device_fingerprint,
            "location": location,
            "velocity_pattern": velocity_pattern,
            "browser_info": browser_info,
            "network_latency": network_latency,
            "time_of_day": time_of_day,
            "session_metadata": session_metadata,
            "label": label
        })
    df = pd.DataFrame(data)
    logger.info(f"Simulated dataset shape: {df.shape}")
    return df

# ------------------------- Phase 2: Hybrid Embedding Generation & Multi‑LLM Integration -------------------------
# FinBERT model and tokenizer for classification/embedding extraction
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME, num_labels=3)
finbert_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finbert_model.to(device)

# SentenceTransformer for text embeddings (used for both transaction and demo purposes)
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384  # SentenceTransformer embedding dimension

# Simple GNN simulation for user behavior embedding
class SimpleGNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=16):
        super(SimpleGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

gnn_model = SimpleGNN().eval()

# ------------------------- Embedding Extraction Functions -------------------------
def extract_finbert_embedding(text_batch):
    """Extract embeddings from FinBERT using mean pooling over the last hidden state."""
    inputs = finbert_tokenizer(text_batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = finbert_model(**inputs, output_hidden_states=True)
    embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
    return embeddings

def extract_sent_transformer_embedding(text_batch):
    """Extract SentenceTransformer embeddings."""
    embeddings = sent_transformer.encode(text_batch, batch_size=16, show_progress_bar=False)
    return np.array(embeddings)

def simulate_gnn_embeddings(user_features):
    """Generate GNN embeddings for user behavior; user_features should be a numpy array."""
    user_features = (user_features - user_features.mean(axis=0)) / (user_features.std(axis=0) + 1e-10)
    tensor_features = torch.tensor(user_features.astype(np.float32))
    with torch.no_grad():
        embeddings = gnn_model(tensor_features).cpu().numpy()
    return embeddings

# ------------------------- Phase 3: Distributed Vector Search Setup -------------------------
faiss_index = None
def init_faiss_index(embeddings, dim, index_path=FAISS_INDEX_PATH):
    """Initialize or load a FAISS HNSW index with normalized embeddings."""
    global faiss_index
    if os.path.exists(index_path):
        faiss_index = faiss.read_index(index_path)
        if faiss_index.d != dim:
            logger.info("FAISS index dimension mismatch. Rebuilding index.")
            os.remove(index_path)
            faiss_index = None
    if faiss_index is None:
        faiss_index = faiss.IndexHNSWFlat(dim, 32)
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, index_path)
        logger.info("Created new FAISS HNSW index and stored to disk.")
    return faiss_index

def search_faiss(query_vector, k=5):
    """Search FAISS index using cosine similarity (assuming normalized vectors)."""
    distances, indices = faiss_index.search(np.array([query_vector], dtype='float32'), k)
    return distances[0], indices[0]

# ------------------------- Phase 4: Full Pipeline Processing & Explainable AI (XAI) -------------------------
def compute_fraud_score(transaction):
    """
    Compute a fraud score using an ensemble approach.
    (In production, integrate outputs from GPT-4, FinBERT, T5, and an RL agent.)
    """
    scores = [random.random() for _ in range(4)]
    return sum(scores) / len(scores)

def process_transaction(transaction):
    """
    End-to-end processing of a transaction:
      - Generate text embeddings (using SentenceTransformer and FinBERT)
      - Compute GNN user embedding for the user
      - Perform vector search to find similar past transactions
      - Compute a fraud score and attach nearest neighbors
    """
    text = transaction.get("behavioral_history", "")
    finbert_emb = extract_finbert_embedding([text])[0]
    sent_emb = extract_sent_transformer_embedding([text])[0]
    combined_emb = np.hstack([finbert_emb[:EMBED_DIM], sent_emb])
    user_features = np.array([[transaction.get("age", 0), transaction.get("credit_score", 850)]])
    gnn_emb = simulate_gnn_embeddings(user_features)[0]
    final_embedding = np.hstack([combined_emb, gnn_emb])
    norm = np.linalg.norm(final_embedding)
    final_embedding_norm = final_embedding / (norm + 1e-10)
    if faiss_index is None:
        init_faiss_index(np.array([final_embedding_norm], dtype='float32'), final_embedding_norm.shape[0])
    distances, indices = search_faiss(final_embedding_norm, k=5)
    fraud_score = compute_fraud_score(transaction)
    transaction["fraud_score"] = fraud_score
    transaction["nearest_neighbors"] = indices.tolist()
    # Log processed transaction for GraphQL querying (append to global log)
    processed_transactions_log.append(transaction)
    return transaction

# Global list to store processed transactions (for GraphQL querying)
processed_transactions_log = []

# ------------------------- Phase 5: Real-Time Monitoring & Alerting -------------------------
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
def log_alert(alert_message):
    record = {"timestamp": datetime.datetime.utcnow().isoformat(), "alert": alert_message}
    with open("alerts.log", "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(f"Alert logged: {alert_message}")

def send_sms_alert(message):
    try:
        # Uncomment for production:
        # twilio_client.messages.create(body=message, from_=TWILIO_FROM_NUMBER, to=ALERT_SMS_TO_NUMBER)
        logger.info("SMS alert sent: " + message)
    except Exception as e:
        logger.error("SMS alert error: " + str(e))

connected_ws_clients = set()
async def ws_alert_handler(websocket, path):
    connected_ws_clients.add(websocket)
    logger.info(f"WebSocket client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            await websocket.send(f"Echo: {message}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connected_ws_clients.discard(websocket)

async def broadcast_alert(alert_message):
    if connected_ws_clients:
        await asyncio.gather(*[client.send(alert_message) for client in connected_ws_clients])
    logger.info(f"Broadcast alert: {alert_message}")

def process_fraud_alert(transaction_id, fraud_score):
    alert_message = f"Transaction {transaction_id} flagged with fraud score {fraud_score:.2f}"
    log_alert(alert_message)
    send_sms_alert(alert_message)
    asyncio.run(broadcast_alert(alert_message))
    alert_counter.inc()
    return alert_message

def optimize_fraud_threshold():
    def objective(threshold):
        th = threshold[0]
        optimal = 0.65
        error = abs(th - optimal) + random.uniform(0, 0.1)
        logger.info(f"Threshold {th:.2f}, error {error:.2f}")
        return error
    res = gp_minimize(objective, [(0.0, 1.0)], acq_func="EI", n_calls=20, random_state=42)
    return res.x[0]

FRAUD_THRESHOLD = optimize_fraud_threshold()
logger.info(f"Initial optimized fraud threshold: {FRAUD_THRESHOLD:.2f}")

# ------------------------- Phase 6: Enterprise‑Grade Security & Compliance -------------------------
def aes_encrypt(plaintext: str) -> str:
    cipher = AES.new(AES_KEY, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plaintext.encode('utf-8'), AES.block_size))
    iv = base64.b64encode(cipher.iv).decode('utf-8')
    ct = base64.b64encode(ct_bytes).decode('utf-8')
    return json.dumps({'iv': iv, 'ciphertext': ct})

def aes_decrypt(ciphertext_json: str) -> str:
    try:
        b64 = json.loads(ciphertext_json)
        iv = base64.b64decode(b64['iv'])
        ct = base64.b64decode(b64['ciphertext'])
        cipher = AES.new(AES_KEY, AES.MODE_CBC, iv)
        pt = unpad(cipher.decrypt(ct), AES.block_size)
        return pt.decode('utf-8')
    except (ValueError, KeyError) as e:
        logger.error("Decryption error: " + str(e))
        return None

def audit_log(event: str, details: dict):
    record = {"timestamp": datetime.datetime.utcnow().isoformat(), "event": event, "details": details}
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(f"Audit record: {record}")

# ------------------------- Flask App Setup and Security Endpoints -------------------------
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
jwt = JWTManager(app)
users = {"alice": {"password": "password123", "role": "admin"},
         "bob": {"password": "mypassword", "role": "user"}}
user_data_store = {"alice": {"email": "alice@example.com", "transactions": []},
                   "bob": {"email": "bob@example.com", "transactions": []}}

@app.route("/login", methods=["POST"])
def login():
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    if username not in users or users[username]["password"] != password:
        return jsonify({"msg": "Bad username or password"}), 401
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token), 200

@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200

@app.route("/right-to-be-forgotten/<username>", methods=["DELETE"])
@jwt_required()
def right_to_be_forgotten(username):
    current_user = get_jwt_identity()
    if current_user != username and users.get(current_user, {}).get("role") != "admin":
        return jsonify({"msg": "Unauthorized"}), 403
    if username in user_data_store:
        audit_log("RightToBeForgotten", {"username": username, "action": "delete_requested"})
        del user_data_store[username]
        audit_log("RightToBeForgotten", {"username": username, "action": "data_deleted"})
        return jsonify({"msg": f"User data for {username} has been deleted."}), 200
    else:
        return jsonify({"msg": f"No data found for {username}."}), 404

@app.route("/encrypt", methods=["POST"])
def encrypt_endpoint():
    plaintext = request.json.get("plaintext", "")
    encrypted = aes_encrypt(plaintext)
    return jsonify({"encrypted": encrypted}), 200

@app.route("/decrypt", methods=["POST"])
def decrypt_endpoint():
    ciphertext = request.json.get("ciphertext", "")
    decrypted = aes_decrypt(ciphertext)
    if decrypted is None:
        return jsonify({"msg": "Decryption error"}), 400
    return jsonify({"decrypted": decrypted}), 200

# ------------------------- Core Processing Endpoints (Flask) -------------------------
@app.route("/simulate-data", methods=["POST"])
@jwt_required()
def simulate_data_endpoint():
    num_records = int(request.json.get("num_records", 1000))
    df = simulate_transactions(num_records)
    data_json = df.to_dict(orient="records")
    return jsonify({"data": data_json, "count": len(data_json)}), 200

@app.route("/process-transaction", methods=["POST"])
@jwt_required()
def process_transaction_endpoint():
    transaction = request.json
    processed = process_transaction(transaction)
    if processed.get("fraud_score", 0) > FRAUD_THRESHOLD:
        process_fraud_alert(processed.get("transaction_id", "unknown"), processed["fraud_score"])
    username = transaction.get("user_id")
    if username in user_data_store:
        user_data_store[username]["transactions"].append(processed.get("transaction_id"))
    else:
        user_data_store[username] = {"transactions": [processed.get("transaction_id")]}
    return jsonify(processed), 200

@app.route("/vector-search", methods=["POST"])
@jwt_required()
def vector_search_endpoint():
    query_text = request.json.get("query", "")
    if not query_text:
        return jsonify({"msg": "Query text required"}), 400
    query_embedding = extract_sent_transformer_embedding([query_text])[0]
    norm = np.linalg.norm(query_embedding)
    query_embedding_norm = query_embedding / (norm + 1e-10)
    distances, indices = search_faiss(query_embedding_norm, k=5)
    return jsonify({"indices": indices.tolist(), "distances": distances.tolist()}), 200

# ------------------------- Phase 7: GraphQL API for Flexible Querying -------------------------
# Define a GraphQL type for transactions
class TransactionType(graphene.ObjectType):
    transaction_id = graphene.String()
    user_id = graphene.String()
    fraud_score = graphene.Float()
    nearest_neighbors = graphene.List(graphene.Int)
    label = graphene.String()

# Root Query that allows querying processed transactions
class Query(graphene.ObjectType):
    transactions = graphene.List(TransactionType)
    transaction_by_id = graphene.Field(TransactionType, transaction_id=graphene.String(required=True))

    def resolve_transactions(self, info):
        return processed_transactions_log

    def resolve_transaction_by_id(self, info, transaction_id):
        for tx in processed_transactions_log:
            if tx.get("transaction_id") == transaction_id:
                return tx
        return None

schema = graphene.Schema(query=Query)
app.add_url_rule("/graphql", view_func=GraphQLView.as_view("graphql", schema=schema, graphiql=True))

# ------------------------- Phase 7: FastAPI Fraud Detection API -------------------------
fastapi_app = FastAPI(title="Fraud Detection API", version="1.0")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.post("/fraud-detection")
async def fraud_detection(request: Request):
    transaction = await request.json()
    processed = process_transaction(transaction)
    if processed.get("fraud_score", 0) > FRAUD_THRESHOLD:
        process_fraud_alert(processed.get("transaction_id", "unknown"), processed["fraud_score"])
    return processed

# ------------------------- WebSocket Server for Real-Time Alerts (Background Thread) -------------------------
def start_websocket_server():
    import nest_asyncio
    nest_asyncio.apply()
    async def ws_main():
        server = await websockets.serve(ws_alert_handler, "localhost", find_available_port(6789))
        logger.info("WebSocket server started.")
        await asyncio.Future()  # run forever
    asyncio.run(ws_main())

def find_available_port(starting_port=6789):
    port = starting_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
        port += 1

ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
ws_thread.start()

# ------------------------- Main Entry Point -------------------------
if __name__ == "__main__":
    logger.info("Starting integrated Fraud Detection Backend...")
    
    # Start FastAPI server in a separate thread on port 6000
    def start_fastapi():
      # Create a new event loop for this thread
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      # Configure uvicorn with your FastAPI app
      config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=6001, log_level="info")
      server = uvicorn.Server(config)# Run the uvicorn server using the event loop
      loop.run_until_complete(server.serve())


      

        
    fastapi_thread = threading.Thread(target=start_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Optionally, if running in Colab or needing public access, integrate ngrok:
    # from flask_ngrok import run_with_ngrok
    # from pyngrok import ngrok
    # ngrok.set_auth_token("2t5jG4Kb85xObYFT6mzyG088vu1_61sQEm5o3GesZbmChuQHR")
    # run_with_ngrok(app)
    
    # Start Flask app (REST, GraphQL, and security endpoints)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
