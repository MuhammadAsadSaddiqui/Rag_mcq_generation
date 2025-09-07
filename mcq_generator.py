import re
import json
import ast
import logging
from typing import List, Dict, Optional

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from langchain.docstore.document import Document
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rag_system import RAGSystem

MODEL_NAME = "asad1575/auto-grad"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
MAX_NEW_TOKENS = 250
TEMPERATURE = 0.6
MIN_P = 0.1
DO_SAMPLE = True
MAX_RETRIES = 3
MIN_QUESTION_LENGTH = 10
MIN_OPTION_LENGTH = 2
MIN_UNIQUE_OPTIONS = 3


logger = logging.getLogger(__name__)
console = Console()


class MCQGenerator:

    def __init__(self, model_name: str = None, subject_domain: str = None):

        self.model_name = model_name or MODEL_NAME
        self.subject_domain = subject_domain
        self.model = None
        self.tokenizer = None
        self.rag_system = None
        self._load_model()

    def __del__(self):
        if hasattr(self, 'rag_system') and self.rag_system:
            self.rag_system.cleanup()

    def _load_model(self):
        logger.info(f"Loading model: {self.model_name}")

        try:
            import torch
            torch.cuda.empty_cache()  # Clear GPU cache first

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=torch.bfloat16,  # Use bfloat16 instead of None
                load_in_4bit=LOAD_IN_4BIT,
                device_map={"": 0},  # Force to GPU 0
                offload_folder="./offload",  # Temporary offload folder
            )

            FastLanguageModel.for_inference(self.model)

            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="llama-3.1",
            )

            logger.info("Model loaded successfully on GPU")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def process_documents(self, documents: List[Document]):
        logger.info("Processing documents for RAG...")

        if self.rag_system:
            self.rag_system.cleanup()

        self.rag_system = RAGSystem()
        self.rag_system.build_index(documents)

    def generate_mcqs(self, num_questions: int, topics: List[str] = None,
                     use_random_context: bool = True) -> List[Dict]:

        logger.info(f"Generating {num_questions} MCQs")

        if not self.rag_system:
            raise ValueError("Documents not processed. Call process_documents() first.")

        mcqs = []

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
        ) as progress:
            task = progress.add_task("Generating MCQs...", total=num_questions)

            for i in range(num_questions):
                success = False

                for retry in range(MAX_RETRIES):
                    try:
                        if use_random_context:
                            # Use random context from documents
                            context = self.rag_system.get_random_context()
                            topic = "general knowledge"
                            logger.info(f"MCQ {i + 1}/{num_questions} - Using random context (Attempt {retry + 1})")
                        else:
                            # Select topic for this question
                            topic = topics[i % len(topics)]
                            # Retrieve relevant context
                            context = self.rag_system.retrieve_relevant_context(topic)
                            logger.info(f"MCQ {i + 1}/{num_questions} - Topic: {topic} (Attempt {retry + 1})")

                        if context and context.strip():
                            logger.info(f"MCQ {i + 1} - Context length: {len(context)}")

                            mcq = self._generate_single_mcq(context, topic)
                            if mcq and self._validate_mcq(mcq):
                                mcqs.append(mcq)
                                logger.info(f"MCQ {i + 1} - Successfully generated and validated")
                                success = True
                                break
                            else:
                                logger.warning(f"Failed to generate valid MCQ {i + 1}, attempt {retry + 1}")
                        else:
                            logger.warning(f"No context retrieved for topic: {topic}")

                    except Exception as e:
                        logger.error(f"Error generating MCQ {i + 1}, attempt {retry + 1}: {e}")
                        continue

                if not success:
                    logger.error(f"Failed to generate MCQ {i + 1} after {MAX_RETRIES} attempts")

                progress.update(task, advance=1)

        logger.info(f"Successfully generated {len(mcqs)} valid MCQs")
        return mcqs

    def _generate_single_mcq(self, context: str, topic: str) -> Optional[Dict]:
        for i in range(5):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates multiple choice questions (MCQ) with options and the correct answer, and you must respond ONLY in strict valid JSON format using double quotes."
                    },
                    {
                        "role": "user",
                        "content": f"Give me an MCQ with correct answer from the given article:\n\n{context}"
                    }
                ]

                # Apply chat template
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda" if torch.cuda.is_available() else "cpu")

                # Generate response
                outputs = self.model.generate(
                    input_ids=inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    temperature=TEMPERATURE,
                    min_p=MIN_P,
                    do_sample=DO_SAMPLE,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract assistant response
                if "<|start_header_id|>assistant<|end_header_id|>" in response:
                    assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
                else:
                    assistant_response = response.split("assistant")[-1].strip() if "assistant" in response else response

                logger.info(f"Raw AI response: {response[:200]}... for content: {context[:100]}...")

                mcq = self._parse_mcq_robust(assistant_response)
                return mcq

            except Exception as e:
                logger.error(f"Error in MCQ generation: {e}")
        return None

    def _parse_mcq_robust(self, response: str) -> Optional[Dict]:
        # Method 1: Try standard JSON parsing
        try:
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                mcq = json.loads(json_str)
                if self._validate_mcq(mcq):
                    logger.info("Parsed with Method 1 (standard JSON)")
                    return mcq
        except Exception as e:
            logger.debug(f"Method 1 failed: {e}")

        # Method 2: Try parsing with single quotes using ast.literal_eval
        try:
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Fix common issues
                json_str = json_str.replace("'", '"')  # Replace single quotes
                mcq = json.loads(json_str)
                if self._validate_mcq(mcq):
                    logger.info("Parsed with Method 2 (quote fixing)")
                    return mcq
        except Exception as e:
            logger.debug(f"Method 2 failed: {e}")

        # Method 3: Try ast.literal_eval for Python-style dicts
        try:
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                mcq = ast.literal_eval(json_str)
                if self._validate_mcq(mcq):
                    logger.info("Parsed with Method 3 (ast.literal_eval)")
                    return mcq
        except Exception as e:
            logger.debug(f"Method 3 failed: {e}")

        # Method 4: Manual parsing as last resort
        try:
            mcq = self._manual_parse_mcq(response)
            if mcq and self._validate_mcq(mcq):
                logger.info("Parsed with Method 4 (manual parsing)")
                return mcq
        except Exception as e:
            logger.debug(f"Method 4 failed: {e}")

        logger.error("All parsing methods failed")
        logger.error(f"Problematic response: {response[:200]}...")
        return None

    def _manual_parse_mcq(self, response: str) -> Optional[Dict]:
        try:
            question_match = re.search(r'["\']?question["\']?\s*:\s*["\']([^"\']+)["\']', response, re.IGNORECASE)
            if not question_match:
                return None
            question = question_match.group(1)

            answer_match = re.search(r'["\']?answer["\']?\s*:\s*["\']([ABCD])["\']', response, re.IGNORECASE)
            if not answer_match:
                return None
            answer = answer_match.group(1)

            options_match = re.search(r'["\']?options["\']?\s*:\s*\[(.*?)\]', response, re.DOTALL | re.IGNORECASE)
            if not options_match:
                return None

            options_str = options_match.group(1)
            options = re.findall(r'["\']([^"\']+)["\']', options_str)

            if len(options) != 4:
                return None

            return {
                "question": question,
                "options": options,
                "answer": answer
            }

        except Exception as e:
            logger.error(f"Manual parsing failed: {e}")
            return None

    def _validate_mcq(self, mcq: Dict) -> bool:
        try:
            if not all(field in mcq for field in ['question', 'options', 'answer']):
                logger.warning("MCQ missing required fields")
                return False

            question = mcq.get('question', '').strip()
            if not question or len(question) < MIN_QUESTION_LENGTH:
                logger.warning("Question too short or empty")
                return False

            placeholder_phrases = [
                "Your question here?", "Question here", "Insert question",
                "helpful assistant", "I am an AI", "I cannot", "I don't know",
                "What is the purpose of this content?", "The article is about"
            ]
            if any(phrase in question for phrase in placeholder_phrases):
                logger.warning("Question contains placeholder or system text")
                return False

            options = mcq.get('options', [])
            if not isinstance(options, list) or len(options) != 4:
                logger.warning(f"Invalid options count: {len(options)}")
                return False

            clean_options = [str(opt).strip() for opt in options]
            if any(not opt or len(opt) < MIN_OPTION_LENGTH for opt in clean_options):
                logger.warning("Found empty or too short options")
                return False

            if len(set(clean_options)) < MIN_UNIQUE_OPTIONS:
                logger.warning("Options are too similar")
                return False

            placeholder_options = ["Option A", "Option B", "Option C", "Option D", "Choice A", "Choice B"]
            if any(opt in clean_options for opt in placeholder_options):
                logger.warning("Found placeholder options")
                return False

            answer = str(mcq.get('answer', '')).strip().upper()
            if answer not in ['A', 'B', 'C', 'D']:
                logger.warning(f"Invalid answer: {answer}")
                return False

            logger.info("MCQ validation passed")
            return True

        except Exception as e:
            logger.error(f"Error validating MCQ: {e}")
            return False

    def set_subject_domain(self, domain: str):
        self.subject_domain = domain
        logger.info(f"Subject domain updated to: {domain}")

    def get_context_stats(self) -> Dict:
        """Get statistics about the loaded context"""
        if not self.rag_system:
            return {"status": "no_documents_loaded"}

        return self.rag_system.get_vectorstore_stats()


def create_programming_mcq_generator(model_name: str = None) -> MCQGenerator:
    return MCQGenerator(model_name=model_name, subject_domain="programming")

def create_science_mcq_generator(model_name: str = None) -> MCQGenerator:
    return MCQGenerator(model_name=model_name, subject_domain="science")

def create_generic_mcq_generator(model_name: str = None) -> MCQGenerator:
    return MCQGenerator(model_name=model_name, subject_domain=None)