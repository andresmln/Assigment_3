"""
Script to create sample evaluation documents for the RAG system.
Generates 3 documents (2 PDF, 1 DOCX) with known content for evaluation.
Run this locally: python eval/create_eval_docs.py
"""
import os

# Try to import required libraries
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")


# ── Document Content ─────────────────────────────────────────────────────────

AI_CONTENT = """Artificial Intelligence: A Comprehensive Overview

Introduction to Artificial Intelligence

Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and natural language understanding. The field was formally founded at a conference at Dartmouth College in 1956.

The Turing Test

Alan Turing proposed the Turing Test in 1950 as a criterion for machine intelligence. In this test, a human evaluator judges natural language conversations between a human and a machine. If the evaluator cannot reliably distinguish the machine from the human, the machine is said to have passed the test. The test remains one of the most discussed concepts in AI philosophy.

Machine Learning

Machine learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction.

Neural Networks

Neural networks are computing systems vaguely inspired by the biological neural networks that make up animal brains. These systems learn to perform tasks by considering examples, generally without being programmed with task-specific rules. They consist of layers of interconnected nodes or neurons that process information using connectionist approaches to computation.

Deep Learning

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural networks, and convolutional neural networks have been applied to fields including computer vision and natural language processing.

Natural Language Processing

Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to enable computers to understand, interpret, and generate human language in a valuable way. Applications include machine translation, sentiment analysis, chatbots, and text summarization.

Transfer Learning

Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second task. It is particularly popular in deep learning where pre-trained models are used as feature extractors or fine-tuned on new datasets, significantly reducing training time and data requirements.

Reinforcement Learning

Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize cumulative reward. The agent learns by interacting with its environment, receiving rewards or penalties for actions taken, and adjusting its strategy accordingly.

Computer Vision

Computer vision is a field of AI that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects and then react to what they see.

Ethics in AI

The development of AI raises important ethical questions about bias, privacy, transparency, and accountability. As AI systems become more prevalent in decision-making processes, ensuring fairness and avoiding discrimination becomes increasingly critical. Researchers and policymakers are working to develop frameworks for responsible AI development.
"""

CLIMATE_CONTENT = """Climate Change: Causes, Effects, and Mitigation Strategies

Introduction

Climate change refers to long-term shifts in global or regional climate patterns. While climate has changed throughout Earth's history, the current rapid warming is primarily attributed to human activities, particularly the burning of fossil fuels and deforestation.

The Greenhouse Effect

The greenhouse effect is a natural phenomenon where greenhouse gases in Earth's atmosphere (including water vapor, carbon dioxide, and methane) trap some of the sun's heat near the surface, making the planet habitable. However, human activities have significantly increased the concentration of these gases, intensifying the greenhouse effect and causing global warming.

Primary Causes of Climate Change

The primary drivers of climate change are greenhouse gas emissions, particularly carbon dioxide from burning fossil fuels like coal, oil, and natural gas. Deforestation contributes by reducing the planet's capacity to absorb CO2. Industrial processes and agriculture also release significant amounts of methane and nitrous oxide.

Global Temperature Rise

Since the pre-industrial era, the average global temperature has risen by approximately 1.1 degrees Celsius. The last decade was the warmest on record. Scientists project that without significant emission reductions, temperatures could rise by 2.5 to 4.5 degrees Celsius by the end of this century.

Sea Level Rise

Rising sea levels pose severe threats to coastal regions worldwide. Effects include increased coastal flooding, accelerated erosion, loss of coastal wetlands and habitats, saltwater contamination of freshwater resources, and the potential displacement of millions of people living in low-lying areas. Current projections suggest sea levels could rise by 0.3 to 1.0 meters by 2100.

Extreme Weather Events

Climate change is increasing the frequency and intensity of extreme weather events, including hurricanes, droughts, heatwaves, and heavy rainfall. These events cause significant economic damage, displace communities, and threaten food and water security worldwide.

The Paris Agreement

The Paris Agreement, adopted in December 2015, is a landmark international accord within the United Nations Framework Convention on Climate Change. Its central aim is to strengthen the global response to climate change by keeping the global temperature rise this century well below 2 degrees Celsius above pre-industrial levels, while pursuing efforts to limit the increase to 1.5 degrees Celsius.

Renewable Energy Solutions

Renewable energy sources including solar photovoltaic, wind turbines, and hydroelectric power generate electricity without directly emitting greenhouse gases during operation. Transitioning from fossil fuels to renewables is one of the most effective strategies for reducing carbon emissions and mitigating climate change.

Carbon Capture and Storage

Carbon capture and storage (CCS) is a technology that captures CO2 emissions from sources like power plants and industrial processes, and stores it underground to prevent it from entering the atmosphere. While promising, the technology is still expensive and not yet deployed at scale.

Individual Actions

Individuals can contribute to climate change mitigation by reducing energy consumption, using public transportation, eating less meat, reducing waste, and supporting renewable energy. While individual actions alone are insufficient, they contribute to broader societal change and market signals.
"""

PYTHON_CONTENT = """Python Programming: A Comprehensive Guide

Introduction to Python

Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its use of significant indentation. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming.

Variables and Data Types

Python supports several built-in data types including integers, floats, strings, booleans, lists, tuples, sets, and dictionaries. Variables in Python are dynamically typed, meaning you don't need to declare their type explicitly. Python uses duck typing, where the type of an object is determined by its behavior rather than its explicit type.

Lists and Tuples

Lists and tuples are both sequence data types in Python. The key difference is mutability: lists are mutable, meaning their elements can be changed, added, or removed after creation using square brackets []. Tuples are immutable, created with parentheses (), and once defined their elements cannot be modified. Tuples are generally faster and use less memory.

List Comprehensions

List comprehensions provide a concise way to create lists in Python. They consist of brackets containing an expression followed by a for clause, then zero or more for or if clauses. The result is a new list resulting from evaluating the expression in the context of the clauses that follow. For example: squares = [x**2 for x in range(10)].

Dictionaries

Dictionaries in Python are unordered collections of key-value pairs. They are created using curly braces {} or the dict() constructor. Keys must be immutable types (strings, numbers, tuples), while values can be any type. Dictionaries provide O(1) average time complexity for lookups.

Functions

Functions in Python are defined using the def keyword. They can accept positional arguments, keyword arguments, default values, and variable-length arguments (*args and **kwargs). Python functions are first-class objects, meaning they can be passed as arguments, returned from other functions, and assigned to variables.

Decorators

Decorators in Python are a powerful design pattern that allows a user to add new functionality to an existing object without modifying its structure. Decorators are usually called before the definition of a function you want to decorate, using the @decorator_name syntax. They wrap the original function and can execute code before and after the wrapped function runs.

Exception Handling

Exception handling in Python uses try/except blocks. The code that might cause an exception is placed in the try block. If an exception occurs, the program jumps to the corresponding except block. You can also use else (executed if no exception occurs) and finally (always executed) clauses. Multiple except blocks can handle different exception types.

The Global Interpreter Lock

The Global Interpreter Lock (GIL) is a mechanism used in CPython to synchronize access to Python objects, preventing multiple native threads from executing Python bytecodes at once. While it simplifies memory management, it can be a bottleneck for CPU-bound multi-threaded programs. Common workarounds include using multiprocessing or alternative Python implementations like Jython.

Object-Oriented Programming

Python supports object-oriented programming with classes and objects. Classes are defined using the class keyword and can include attributes (data) and methods (functions). Python supports inheritance, polymorphism, and encapsulation. The __init__ method serves as the constructor, and self refers to the instance being created.

Generators and Iterators

Generators are a special type of function that return an iterator using the yield keyword instead of return. They generate values lazily, one at a time, which is memory-efficient for large datasets. Generator expressions provide a compact syntax similar to list comprehensions but produce values on demand.

Virtual Environments

Virtual environments in Python allow you to create isolated Python environments for different projects. Each environment can have its own set of installed packages, independent of the system Python installation. The venv module (built-in since Python 3.3) and tools like virtualenv and conda are commonly used to manage virtual environments.
"""


def create_pdfs():
    """Create sample PDF documents using reportlab."""
    if not HAS_REPORTLAB:
        print("reportlab not installed. Install with: pip install reportlab")
        print("Creating text files instead...")
        # Fallback: create text files that can manually be converted
        with open(os.path.join(OUTPUT_DIR, "artificial_intelligence_overview.txt"), "w") as f:
            f.write(AI_CONTENT)
        with open(os.path.join(OUTPUT_DIR, "climate_change_report.txt"), "w") as f:
            f.write(CLIMATE_CONTENT)
        return

    styles = getSampleStyleSheet()

    # AI Document
    pdf_path = os.path.join(OUTPUT_DIR, "artificial_intelligence_overview.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []
    for para in AI_CONTENT.strip().split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), styles["Normal"]))
            story.append(Spacer(1, 12))
    doc.build(story)
    print(f"Created: {pdf_path}")

    # Climate Document
    pdf_path = os.path.join(OUTPUT_DIR, "climate_change_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []
    for para in CLIMATE_CONTENT.strip().split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), styles["Normal"]))
            story.append(Spacer(1, 12))
    doc.build(story)
    print(f"Created: {pdf_path}")


def create_docx():
    """Create sample DOCX document."""
    if not HAS_DOCX:
        print("python-docx not installed. Install with: pip install python-docx")
        with open(os.path.join(OUTPUT_DIR, "python_programming_guide.txt"), "w") as f:
            f.write(PYTHON_CONTENT)
        return

    doc = Document()
    paragraphs = PYTHON_CONTENT.strip().split("\n\n")
    for para in paragraphs:
        if para.strip():
            # First line of the content is the title
            doc.add_paragraph(para.strip())

    docx_path = os.path.join(OUTPUT_DIR, "python_programming_guide.docx")
    doc.save(docx_path)
    print(f"Created: {docx_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Creating sample evaluation documents...")
    create_pdfs()
    create_docx()
    print("Done! Documents saved to:", OUTPUT_DIR)
