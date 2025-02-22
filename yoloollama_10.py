# This code takes uer prompt about an image url or local drive path. The local LLM via Ollama extracts the image ULR or path from the query. Then the URL or path is passed to the Yolo model for object detection. The detected objects are then sent back to the LLM. The LLM reponds back with a simple summarized text about what is in that image.
# yolo 
# ollama
# Partha Pratim Ray, 2025

"""
Usage:

curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Here is my image: /home/pi/Desktop/yoloollama/cat1.jpg. Please analyze it!"}'
"""

import os
import csv
import json
import time
import requests
from flask import Flask, request, jsonify
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)

# ------------------------------------------------------------------------
# 1) Configuration
# ------------------------------------------------------------------------
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# LLM models
LLM_EXTRACTOR_NAME = "qwen2.5:0.5b-instruct"
LLM_SUMMARY_NAME   = "granite3-moe:1b-instruct-q4_K_M"

# YOLO model file name
YOLO_MODEL_PATH = "yolo8s.pt"

# CSV logging
CSV_FILE = "metrics_log.csv"
CSV_HEADERS = [
    "timestamp",
    "user_prompt",

    # LLM #1 (extraction)
    "extraction_total_duration_ns",
    "extraction_load_duration_ns",
    "extraction_prompt_eval_count",
    "extraction_prompt_eval_duration_ns",
    "extraction_eval_count",
    "extraction_eval_duration_ns",
    "extraction_tokens_per_second",
    "json_extraction_duration_ns",

    # YOLO
    "yolo_inference_ns",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",

    # LLM #2 (summary)
    "summary_total_duration_ns",
    "summary_load_duration_ns",
    "summary_prompt_eval_count",
    "summary_prompt_eval_duration_ns",
    "summary_eval_count",
    "summary_eval_duration_ns",
    "summary_tokens_per_second",

    # LLM responses
    "extraction_llm_response",
    "summary_llm_response",
    
    # New columns for LLM model names
    "llm_extraction_model_name",
    "llm_summary_model_name",
    
    # Yolo model name
    "yolo_model_name"
]

# ------------------------------------------------------------------------
# 2) Load YOLO Model (PyTorch-based)
# ------------------------------------------------------------------------
try:
    model = YOLO(YOLO_MODEL_PATH)  # or your local .pt model
    print("[DEBUG] YOLO model loaded successfully.")
except Exception as e:
    print("[ERROR] Failed to load YOLO model:", e)
    model = None

# ------------------------------------------------------------------------
# 3) CSV logging function
# ------------------------------------------------------------------------
def log_metrics_to_csv(row_data):
    """
    Append row_data to CSV_FILE. Create headers if file does not exist.
    """
    try:
        file_exists = os.path.isfile(CSV_FILE)
        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                print(f"[DEBUG] Creating CSV file '{CSV_FILE}' and writing headers.")
                writer.writerow(CSV_HEADERS)
            writer.writerow(row_data)
        print("[DEBUG] Appended row to CSV:", row_data)
    except Exception as e:
        print("[ERROR] Could not write CSV:", e)

# ------------------------------------------------------------------------
# 4) LLM #1: Extract the URL
# ------------------------------------------------------------------------
def build_extraction_prompt(user_prompt: str) -> str:
    multi_shot_examples = """
You are given a user prompt. Your job:
1. Find exactly one image link/path/base64 from the user prompt.
2. Return valid JSON with one key: "extracted_url".
3. Do NOT rewrite or shorten the path/URL; copy it EXACTLY from the user prompt if found.
4. If no path or URL is found, "extracted_url" should be "" (empty string).

Multi-shot examples:

Example 1:
User Prompt: "My image is /home/pi/Desktop/images/dog.png. Please analyze it!"
Assistant:
{
  "extracted_url": "/home/pi/Desktop/images/dog.png"
}

Example 2:
User Prompt: "Check this link: https://mysite.com/cat.jpg"
Assistant:
{
  "extracted_url": "https://mysite.com/cat.jpg"
}

Example 3:
User Prompt: "Here is base64 => data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQ..."
Assistant:
{
  "extracted_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQ..."
}

Example 4:
User Prompt: "No images here, just text..."
Assistant:
{
  "extracted_url": ""
}

Now, extract the link/path from this user prompt:
""".strip()
    return f"""{multi_shot_examples}
User Prompt: "{user_prompt}"
Assistant:
"""

def call_llm_extractor(prompt_text: str):
    """
    Calls Ollama with structured output:
    { "extracted_url": "<string>" }
    """
    schema_format = {
        "type": "object",
        "properties": {
            "extracted_url": {"type": "string"}
        },
        "required": ["extracted_url"]
    }
    # Determine keep_alive based on whether the two models are different.
    extraction_keep_alive = 0 if LLM_EXTRACTOR_NAME != LLM_SUMMARY_NAME else -1
    payload = {
        "model": LLM_EXTRACTOR_NAME,
        "keep_alive": extraction_keep_alive,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "format": schema_format,
        "options": {
            "temperature": 0.0
        }
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload)
    resp.raise_for_status()
    return resp.json()

# ------------------------------------------------------------------------
# 5) YOLO detection
# ------------------------------------------------------------------------
def run_yolo_detection(image_path_or_url: str):
    """
    In the latest Ultralytics, iterate over results[0].boxes, then read speed from results[0].speed.
    """
    if model is None:
        raise RuntimeError("YOLO model not loaded.")

    results = model.predict(source=image_path_or_url)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        class_name = results[0].names.get(cls_id, f"class_{cls_id}")
        detections.append((class_name, conf))

    # Speed info in ms
    speed_info = results[0].speed  # dict with keys: 'preprocess', 'inference', 'postprocess'
    preprocess_ms = speed_info.get("preprocess", 0.0)
    inference_ms  = speed_info.get("inference", 0.0)
    postprocess_ms= speed_info.get("postprocess", 0.0)

    return detections, preprocess_ms, inference_ms, postprocess_ms

# ------------------------------------------------------------------------
# 6) LLM #2: Summarize (short)
# ------------------------------------------------------------------------
def build_summary_prompt(detections):
    if not detections:
        return "No objects detected in the image. Summarize with minimal words."
    items_str = ", ".join([f"{c}({conf:.2f})" for (c, conf) in detections])
    return f"Detected objects: {items_str}. Write a very short summary with minimal words."

def call_llm_summary(prompt_text: str):
    # Determine keep_alive based on whether the two models are different.
    summary_keep_alive = 0 if LLM_EXTRACTOR_NAME != LLM_SUMMARY_NAME else -1
    payload = {
        "model": LLM_SUMMARY_NAME,
        "keep_alive": summary_keep_alive,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload)
    resp.raise_for_status()
    return resp.json()

# ------------------------------------------------------------------------
# 7) /detect route
# ------------------------------------------------------------------------
@app.route("/detect", methods=["POST"])
def detect_image():
    """
    1) Extract link from user prompt.
    2) YOLO detection (with speed in ms).
    3) LLM summary (brief).
    4) CSV logging with YOLO speed columns and LLM responses.
    """
    req_data = request.get_json(force=True)
    user_prompt = req_data.get("prompt", "")
    print("[DEBUG] Received user prompt:", user_prompt)

    # LLM #1: Extraction
    extraction_prompt = build_extraction_prompt(user_prompt)
    try:
        extract_resp = call_llm_extractor(extraction_prompt)
    except Exception as e:
        return jsonify({"error": f"LLM extraction call failed: {e}"}), 500

    # LLM #1 metrics
    ext_total_duration_ns = extract_resp.get("total_duration", 0)
    ext_load_duration_ns = extract_resp.get("load_duration", 0)
    ext_prompt_eval_count = extract_resp.get("prompt_eval_count", 0)
    ext_prompt_eval_duration_ns = extract_resp.get("prompt_eval_duration", 0)
    ext_eval_count = extract_resp.get("eval_count", 0)
    ext_eval_duration_ns = extract_resp.get("eval_duration", 1)
    ext_tokens_per_second = 0.0
    if ext_eval_duration_ns > 0:
        ext_tokens_per_second = ext_eval_count / ext_eval_duration_ns * 1e9

    # LLM #1 raw text
    extraction_llm_response = extract_resp.get("message", {}).get("content", "")

    # JSON parse
    parse_start_ns = time.time_ns()
    try:
        extracted_obj = json.loads(extraction_llm_response)
        extracted_url = extracted_obj.get("extracted_url", "").strip()
    except json.JSONDecodeError:
        extracted_url = ""
    parse_end_ns = time.time_ns()
    json_extraction_duration_ns = parse_end_ns - parse_start_ns

    print("[DEBUG] Extracted URL from LLM:", extracted_url)
    if not extracted_url:
        return jsonify({"status": "no_image", "message": "No valid image link extracted."})

    # YOLO
    yolo_start = time.time()
    try:
        detections, pre_ms, inf_ms, post_ms = run_yolo_detection(extracted_url)
    except Exception as e:
        return jsonify({"error": f"YOLO detection failed: {e}"}), 500
    yolo_end = time.time()
    yolo_inference_ns = int((yolo_end - yolo_start) * 1e9)

    # LLM #2: Summarize
    summary_prompt = build_summary_prompt(detections)
    try:
        summary_resp = call_llm_summary(summary_prompt)
    except Exception as e:
        return jsonify({"error": f"LLM summary call failed: {e}"}), 500

    # LLM #2 metrics
    sum_total_duration_ns = summary_resp.get("total_duration", 0)
    sum_load_duration_ns   = summary_resp.get("load_duration", 0)
    sum_prompt_eval_count  = summary_resp.get("prompt_eval_count", 0)
    sum_prompt_eval_duration_ns = summary_resp.get("prompt_eval_duration", 0)
    sum_eval_count  = summary_resp.get("eval_count", 0)
    sum_eval_duration_ns = summary_resp.get("eval_duration", 1)
    sum_tokens_per_second = 0.0
    if sum_eval_duration_ns > 0:
        sum_tokens_per_second = sum_eval_count / sum_eval_duration_ns * 1e9

    # LLM #2 raw text (final summary)
    summary_llm_response = summary_resp.get("message", {}).get("content", "")

    # Build CSV row
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_row = [
        timestamp,
        user_prompt,

        # Extraction LLM metrics
        ext_total_duration_ns,
        ext_load_duration_ns,
        ext_prompt_eval_count,
        ext_prompt_eval_duration_ns,
        ext_eval_count,
        ext_eval_duration_ns,
        round(ext_tokens_per_second, 2),
        json_extraction_duration_ns,

        # YOLO
        yolo_inference_ns,
        pre_ms,
        inf_ms,
        post_ms,

        # Summary LLM metrics
        sum_total_duration_ns,
        sum_load_duration_ns,
        sum_prompt_eval_count,
        sum_prompt_eval_duration_ns,
        sum_eval_count,
        sum_eval_duration_ns,
        round(sum_tokens_per_second, 2),

        # LLM responses
        extraction_llm_response,
        summary_llm_response,
        
        # New columns for LLM model names
        LLM_EXTRACTOR_NAME,
        LLM_SUMMARY_NAME,
        
        # Yolo model name (using the file name)
        YOLO_MODEL_PATH
    ]
    log_metrics_to_csv(csv_row)

    # Return final JSON
    return jsonify({
        "extracted_url": extracted_url,
        "detections": [
            {"class_name": c, "confidence": conf} for (c, conf) in detections
        ],
        "speed_ms": {
            "preprocess": pre_ms,
            "inference": inf_ms,
            "postprocess": post_ms
        },
        "summary_paragraph": summary_llm_response
    })

# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
if __name__ == "__main__":
    print("[DEBUG] Starting Flask server on :5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True)
