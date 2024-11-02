# backend/app/ai_models.py

import logging
from typing import List
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

import openai
from transformers import pipeline

# LangChain imports
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

class HuggingFaceModel:
    def __init__(self, task: str, model_name: str):
        self.pipeline = pipeline(task, model=model_name)

    def generate_summary(self, text: str, max_length: int = 150, min_length: int = 40, do_sample: bool = False) -> str:
        try:
            summary = self.pipeline(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Hugging Face Summarization Error: {e}")
            return "Error generating summary."

    def analyze_sentiment(self, text: str) -> dict:
        try:
            sentiment = self.pipeline(text)
            # Assuming binary sentiment for simplicity
            positive = 0
            negative = 0
            for result in sentiment:
                if result['label'] == 'POSITIVE':
                    positive += result['score']
                else:
                    negative += result['score']
            neutral = 1 - (positive + negative)
            return {
                'positive': round(positive, 2),
                'neutral': round(neutral, 2),
                'negative': round(negative, 2)
            }
        except Exception as e:
            logger.error(f"Hugging Face Sentiment Analysis Error: {e}")
            return {'positive': 0, 'neutral': 0, 'negative': 0}

class OpenAIModel:
    def __init__(self, api_key: str, task: str):
        self.api_key = api_key
        self.task = task
        self.client = openai  # Use the openai module directly
        self.client.api_key = self.api_key
        # Define the paraphrase prompt template
        self.paraphrase_prompt = PromptTemplate(
            input_variables=["text"],
            template="Paraphrase the following text:\n\n{text}"
        )

        # Define the question generation prompt template
        self.question_generation_prompt = PromptTemplate(
            input_variables=["text"],
            template="Generate a list of insightful questions based on the following text:\n\n{text}"
        )

        # Define the answer generation prompt template
        # self.answer_generation_prompt = PromptTemplate(
        #     input_variables=["question", "context"],
        #     template=(
        #         "You are a knowledgeable assistant. Using the context provided, "
        #         "please answer the following question in a clear and concise manner.\n\n"
        #         "Question: {question}\n\n"
        #         "Context: {context}\n\n"
        #         "Answer:"
        #     )
        # )

        self.answer_generation_prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "You are a knowledgeable assistant. Using the context provided, "
                "please answer the following question in a clear and concise manner.\n\n"
                "{text}\n\n"
                "Answer:"
        )
)
        self.llm = ChatOpenAI(openai_api_key=self.api_key, model_name="gpt-3.5-turbo")

    def paraphrase(self, text: str) -> str:
        try:
            prompt = self.paraphrase_prompt.format(text=text)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
            )
            # Corrected attribute access
            paraphrased = response.choices[0].message.content.strip()
            return paraphrased
        except openai.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            pass
        except openai.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass

    def generate_questions(self, text: str) -> List[str]:
        try:
            prompt = self.question_generation_prompt.format(text=text)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                max_tokens=300,
                temperature=0.7,
            )
            # Corrected attribute access
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            # Handle bullet points or numbered lists
            if all(q.startswith(('- ', '* ', '1.', '2.', '3.')) for q in questions):
                questions = [q.lstrip('-*1234567890. ').strip() for q in questions]
            return questions if questions else ["No questions generated."]
        except openai.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            pass
        except openai.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass

    def answer_question(self, question: str, context: str) -> str:
        try:
            # Split the context into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_text(context)
            
            # Create documents with 'source' metadata
            docs = [Document(page_content=t, metadata={"source": f"chunk_{i}"}) for i, t in enumerate(texts)]
            
            # Create embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            vector_store = FAISS.from_documents(docs, embeddings)
            
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 documents

            # Adjust the prompt template to accept 'summaries' and 'question'
            answer_generation_prompt = PromptTemplate(
                input_variables=["summaries", "question"],
                template=(
                    "You are a knowledgeable assistant. Using the context provided, "
                    "please answer the following question in a clear and concise manner.\n\n"
                    "Context: {summaries}\n\n"
                    "Question: {question}\n\n"
                    "Answer:"
                )
            )
            
            # Initialize RetrievalQAWithSourcesChain using from_llm
            qa_chain = RetrievalQAWithSourcesChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                verbose=True,
                return_source_documents=True,
                combine_prompt=answer_generation_prompt  # Use 'combine_prompt' instead of 'prompt'
            )
            # Get answer using the chain
            result = qa_chain({"question": question})
            answer = result.get("answer", "No answer found.")
            
            return answer.strip()
        except openai.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            pass
        except openai.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
