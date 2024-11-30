import os
import json
import requests
from datetime import datetime
import time

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "gemma2:27b"

def call_ollama(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        response.raise_for_status()
        full_response = ""
        for line in response.iter_lines():
            if line:
                line_content = json.loads(line.decode('utf-8'))
                full_response += line_content.get("response", "")
                if line_content.get("done"):
                    break
        return full_response.strip()
    except requests.exceptions.RequestException as e:
        print(f"Ollama API call failed: {e}")
        return None

def analyze_esg_data(company_data: dict) -> str:
    """Analyze the company's ESG data and provide a summary with scores."""
    ESG_PROMPT = f"""
    You are an expert in financial analysis and ESG report analysis. You have been provided with the company's Environmental, Social, and Governance (ESG) data. Your task is to:

    1. **Analyze the Data:** Carefully review the provided ESG data to understand the company's performance in each area.
    2. **Provide a Brief Summary:** Craft a concise summary that highlights the key points from the Environmental, Social, and Governance aspects of the company's operations.
    3. **Assign Scores:** Based on your analysis, assign a score out of 10 for each of the three categories (Environmental, Social, Governance) and an overall ESG score out of 10. Provide a brief justification for each score. Only provide the score for Environmental, Social, Governance and Total. Something like: Environmental: 6/10, Social: 7/10, Governance: 8/10, Total: 7/10

    Please ensure that your analysis is unbiased, fact-based, and suitable for stakeholders interested in an overview of the company's ESG performance. Ensure the summary is concise, highlighting the most significant points from each section without unnecessary elaboration.

    The company's ESG data is as follows:

    **Environmental Data:**
    {company_data["e"]}

    **Social Data:**
    {company_data["s"]}

    **Governance Data:**
    {company_data["g"]}

    Please provide the summary and the ESG score accordingly.
    Example of how the ESG Score should look:
    **ESG Score:**
    Environmental: 6/10 - Limited data available but demonstrates commitment to diversity and inclusion
    Social: 7/10 - Strong framework for social initiatives, emphasis on measurable outcomes needed
    Governance: 8/10 - Robust governance structure, clear integration of ESG into decision-making
    Total: 7/10
    """
    return call_ollama(ESG_PROMPT)

# Main function to handle ESG analysis
def esg_analysis(company_data):
    start_time = time.time()
    # Analyze the ESG data using a single agent
    report = analyze_esg_data(company_data)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    return report

# Example usage
if __name__ == "__main__":
    with open('data.json', 'r') as file:
        esg_data = json.load(file)

    for company, data in esg_data.items():
        print(f"ESG Report for {company}:")
        report = esg_analysis(data)
        today_date = datetime.now().strftime("%Y-%m-%d")
        file_name = f"{company}_esg_summary_{today_date}.txt"
        folder_path = f"./reports/single-agent/{today_date}/{company}"
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "w") as file:
            file.write(report)
        print(f"Report saved to {file_path}")