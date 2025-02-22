import requests
import json

# URL for the Flask server
SERVER_URL = "http://localhost:5000/detect"
# Base path for images
IMAGE_BASE_PATH = "/home/pi/Desktop/yoloollama/"

def main():
    for i in range(5):
        image_file = f"{i}.jpg"
        image_path = IMAGE_BASE_PATH + image_file
        prompt = f"Here is my image: {image_path}. Please analyze it!"
        payload = {"prompt": prompt}

        try:
            response = requests.post(SERVER_URL, json=payload)
            print(f"Response for {image_file}:")
            if response.ok:
                result = response.json()
                print(json.dumps(result, indent=2))
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"An error occurred for {image_file}: {e}")

if __name__ == "__main__":
    main()
