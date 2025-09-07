from temporalio import activity
import json
import requests
from datetime import datetime

mcq_generator = None


def get_mcq_generator():
    global mcq_generator
    if mcq_generator is None:
        from mcq_generator import MCQGenerator
        mcq_generator = MCQGenerator()
    return mcq_generator


@activity.defn
async def generate_mcq_activity(input_data):
    url = input_data.get('url')
    num_questions = input_data.get('num_questions', 5)
    mcq_set_id = input_data.get('mcq_set_id')
    webhook_url = input_data.get('webhook_url')

    print(f"Starting MCQ generation for set {mcq_set_id} with {num_questions} questions")

    if not url:
        if webhook_url and mcq_set_id:
            send_webhook(webhook_url, mcq_set_id, None, False, "URL is required")
        return {"error": "URL is required"}

    if not mcq_set_id or not webhook_url:
        return {"error": "MCQ Set ID and webhook URL are required"}

    try:
        from document_downloader import DocumentDownloader
        from document_processor import DocumentProcessor

        print(f"Downloading document from: {url}")
        downloader = DocumentDownloader()
        doc_processor = DocumentProcessor()

        downloaded_file = downloader.download_document(url)
        print(f"Downloaded file: {downloaded_file}")

        if downloaded_file.suffix.lower() == ".pdf":
            documents = doc_processor.read_pdf(downloaded_file)
        elif downloaded_file.suffix.lower() == ".pptx":
            documents = doc_processor.read_pptx(downloaded_file)
        else:
            error_msg = "Unsupported file format"
            send_webhook(webhook_url, mcq_set_id, None, False, error_msg)
            return {"error": error_msg}

        if not documents:
            error_msg = "Failed to process document or no content found"
            send_webhook(webhook_url, mcq_set_id, None, False, error_msg)
            return {"error": error_msg}

        print(f"Processed {len(documents)} document sections")

        mcq_gen = get_mcq_generator()
        mcq_gen.process_documents(documents)

        print(f"Generating {num_questions} MCQs...")
        mcqs = mcq_gen.generate_mcqs(num_questions)

        # Clean up downloaded file
        downloaded_file.unlink(missing_ok=True)

        if mcqs and len(mcqs) > 0:
            print(f"Successfully generated {len(mcqs)} MCQs")

            # Send success webhook
            send_webhook(webhook_url, mcq_set_id, mcqs, True)

            return {
                "success": True,
                "mcqs": mcqs,
                "total_questions": len(mcqs),
                "mcq_set_id": mcq_set_id
            }
        else:
            error_msg = "Failed to generate any MCQs from the content"
            print(f"Error: {error_msg}")
            send_webhook(webhook_url, mcq_set_id, None, False, error_msg)
            return {"error": error_msg}

    except Exception as e:
        error_msg = f"Activity failed: {str(e)}"
        print(f"Exception in MCQ generation: {error_msg}")
        if webhook_url and mcq_set_id:
            send_webhook(webhook_url, mcq_set_id, None, False, error_msg)
        return {"error": error_msg}


def send_webhook(webhook_url, mcq_set_id, mcqs, success, error=None):
    """Send webhook notification to Next.js API"""
    try:
        payload = {
            "mcq_set_id": mcq_set_id,
            "success": success
        }

        if success and mcqs:
            payload["mcqs"] = mcqs

        if error:
            payload["error"] = error

        print(f"Sending webhook to {webhook_url} for MCQ set {mcq_set_id}")
        print(f"Payload summary: success={success}, mcqs_count={len(mcqs) if mcqs else 0}")

        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            print(f"Webhook sent successfully for MCQ set {mcq_set_id}")
            response_data = response.json()
            print(f"Webhook response: {response_data}")
        else:
            print(f"Webhook failed with status {response.status_code} for MCQ set {mcq_set_id}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"Error sending webhook for MCQ set {mcq_set_id}: {str(e)}")