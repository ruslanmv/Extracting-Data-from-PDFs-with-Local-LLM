## Extracting Data from PDFs with Llama 3.2 Locally: A Complete Example

In this blog, we’ll explore how to build a PDF data extraction pipeline using **Llama 3.2**, an advanced, multilingual large language model (LLM) by Meta, running locally on your machine. This guide covers setting up the model, quantizing it for efficient use on limited hardware, and building an extraction pipeline with a complete example PDF and expected output.

### Why Llama 3.2 and Local Deployment?

The Llama 3.2 model family, including **Llama-3.2-3B-Instruct**, offers robust performance across various languages and complex NLP tasks. Here’s why using it locally is beneficial:
- **Multilingual and Versatile**: Llama 3.2 supports multiple languages and can handle complex tasks, such as agentic retrieval and summarization.
- **Optimized for Local Use**: Llama 3.2 is designed for both cloud and on-device use, with quantization options to fit limited hardware setups.
- **Data Privacy**: Local processing ensures sensitive data stays on your device.
- **Cost-Effective**: Eliminates the need for API costs or cloud infrastructure fees, allowing unlimited, unrestricted processing.

### Model Overview

**Llama 3.2** is an auto-regressive transformer model trained on a mix of publicly available data, with parameters optimized for multilingual tasks. It comes in several configurations, including quantized models, which reduce memory usage and improve computational efficiency on resource-constrained devices.

---

### Key Terms

- **Quantization**: Reduces memory usage by storing model weights in lower precision, such as 4-bit, allowing efficient operation on devices with limited memory.
- **Tokenization**: The process of breaking down text into smaller, manageable units (tokens) for the model to process.
- **Prompt Engineering**: Crafting specific prompts to tailor the model’s output to the desired response.
- **Fine-Tuning**: Adapting a pre-trained model to a specific dataset or task for improved performance.

### Pipeline Objectives

To create a Python-based pipeline that can:
1. **Preprocess PDFs**: Convert PDFs into a format suitable for Llama 3.2.
2. **Run Llama 3.2 Locally**: Efficiently deploy and operate the model on your local device, optimized with quantization.
3. **Extract Key Information**: Identify and extract company names and activities from PDF content.

---

### Sample PDF (`example.pdf`)

To test the pipeline, use the following text in a sample PDF named `example.pdf`:

```
Acme Corp is a leading technology company specializing in artificial intelligence and machine learning solutions. Based in San Francisco, Acme Corp develops cutting-edge software for data analysis, natural language processing, and computer vision.

Beta Industries is a global manufacturing company headquartered in New York City. They are involved in the production of sustainable materials and renewable energy technologies. Beta Industries is committed to environmental responsibility and innovation.
```

---

### Step 1: Setting Up Your Environment

#### Hardware Requirements
- **GPU**: Preferably an NVIDIA GPU with a substantial VRAM allocation for optimal performance.
- **CUDA Toolkit**: Ensure compatibility between CUDA, your GPU, and PyTorch.

#### Software Installation
Install the necessary libraries:

```bash
pip install torch transformers accelerate bitsandbytes PyPDF2
```

#### Model Download

To access the Llama 3.2 model, use the Hugging Face Model Hub:

```bash
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --include "original/*" --local-dir Llama-3.2-3B-Instruct
```

---

### Step 2: Quantizing Llama 3.2

For running Llama 3.2 on devices with limited VRAM, 4-bit quantization is highly recommended. Here’s how to set up quantization using the `bitsandbytes` library.

```python
from transformers import BitsAndBytesConfig

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
```

---

### Step 3: Loading and Fine-Tuning Llama 3.2

Once downloaded, load the quantized model for use. Fine-tuning is optional but enhances accuracy for specific tasks, such as document data extraction.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize model with empty weights for quantized loading
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=True,
        quantization_config=bnb_config
    )

# Load model weights and dispatch to the appropriate device
model = load_checkpoint_and_dispatch(model, model_id, device_map="auto")
```

---

### Step 4: Preprocessing PDFs

Prepare PDFs by converting them into clean, structured text data for the model. This function reads PDF content and removes extraneous whitespace for easier model processing.

```python
import PyPDF2

def preprocess_pdf(pdf_path):
    """
    Extracts text from a PDF and performs basic cleaning.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        cleaned_text: A string containing the extracted and cleaned text.
    """
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    cleaned_text = " ".join(text.split())  # Remove extra whitespace
    return cleaned_text
```

---

### Step 5: Interacting with Llama 3.2

To generate responses from Llama 3.2, define a function to format the text and send it to the model. This function incorporates prompt engineering to customize outputs.

```python
def get_llama_response(text, prompt_template):
    """
    Sends a request to the local Llama 3.2 model and returns the response.

    Args:
        text: The text extracted from the PDF.
        prompt_template: A template for the prompt, including placeholders for the text.

    Returns:
        response_text: The text generated by Llama 3.2.
    """
    prompt = prompt_template.format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text
```

---

### Step 6: Extracting Information

Using tailored prompts, extract specific information such as company names and activities from the Llama 3.2 output.

```python
import json

def extract_information(llm_response):
    """
    Extracts company names and activities from the LLM response.

    Args:
        llm_response: The text generated by the LLM.

    Returns:
        extracted_data: A dictionary containing the extracted information.
    """
    prompt_template = """
    Identify the companies mentioned in this text and their main activities:
    {text}

    Provide your answer in the following JSON format:
    {{"companies": [
        {{"name": "company name 1", "activity": "main activity of company 1"}},
        {{"name": "company name 2", "activity": "main activity of company 2"}},
        ...
    ]}}
    """
    response = get_llama_response(llm_response, prompt_template)
    try:
        data = json.loads(response)
        return data
    except json.JSONDecodeError:
        print("Error: Invalid JSON response from LLM.")
        return None
```

---

### Step 7: Putting it All Together

With the functions defined, you can build a complete pipeline for PDF processing, extracting key data.

```python
def process_pdf(pdf_path):
    """
    Processes a PDF to extract company names and activities.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        extracted_data: A dictionary containing the extracted information.
    """
    text = preprocess_pdf(pdf_path)
    llm_response = get_llama_response(text, prompt_template)
    extracted_data = extract_information(llm_response)
    return extracted_data

# Example usage
pdf_path = "example.pdf"  # Path to the sample PDF
extracted_data = process_pdf(pdf_path)
print(extracted_data)
```

---

### Expected Output

After running the code, you should see an output similar to the following JSON format:

```json
{
  "companies": [
    {
      "name": "Acme Corp",
      "activity": "technology company specializing in artificial intelligence and machine learning solutions"
    },
    {
      "name": "Beta Industries",
      "activity": "global manufacturing company involved in the production of sustainable materials and renewable energy technologies"
    }
  ]
}
```

---

### Running the Example

1. **Save the Code**: Save the code above as a Python file (e.g., `pdf_extractor.py`).
2. **Create `example.pdf`**: Create a PDF file named `example.pdf` with the sample text provided.
3. **Download Llama 3.2**: Download the `meta-llama/Llama-3.2-3B-Instruct` model from Hugging Face.
4. **Update Paths**: In the code, update `"path/to/llama-3.2-3B-Instruct"` with the actual path to your

 downloaded Llama 3.2 model.
5. **Run the Script**: Execute the script from your terminal: `python pdf_extractor.py`

---

### Important Considerations

- **Fine-Tuning**: Fine-tune Llama 3.2 on domain-specific data to improve extraction accuracy.
- **Prompt Engineering**: Experiment with prompts to refine the model’s responses.
- **Error Handling**: Implement error handling to manage unexpected inputs and outputs.
- **Resource Monitoring**: Keep an eye on GPU memory to avoid overloading during processing.

### Conclusion

This guide offers a comprehensive approach to deploying a PDF data extraction pipeline with Llama 3.2 on your local GPU. Ideal for scenarios requiring privacy, flexibility, and scalability, this setup demonstrates how Llama 3.2’s multilingual and instruction-tuned capabilities can be leveraged for extracting structured data from complex documents.

With further fine-tuning and optimization, this method can be expanded to handle various document types and large-scale applications, providing a robust local solution for data extraction.
