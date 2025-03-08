from setuptools import setup, find_packages

setup(
    name="vatsalpatel18_rag_llm_metabolomics",
    version="0.1.0",
    description="A tool for RAG-LLM performance analysis on research papers (metabolomics) with support for GPU and CPU modes.",
    author="Vatsal Patel",
    packages=find_packages(),
    install_requires=[
         "torch",
         "faiss-cpu",
         "PyPDF2",
         "transformers",
         "numpy",
         "openai",
         "python-dotenv",
         "gradio",
         "nltk",
         "rouge-score",
         "pandas",
         "scipy",
         "statsmodels"
    ],
)
