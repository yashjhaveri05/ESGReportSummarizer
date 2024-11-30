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

# Define tools for environmental, social, and governance analyses
def analyze_environmental_data(e_data: str) -> str:
    """Analyze the environmental aspect of the company's ESG data."""
    # prompt = f"Analyze the following environmental data: {e_data}"
    ENV_PROMPT = f"""
        You are an expert in environmental sustainability and analysis. You have been provided with an ESG report, focusing on the Environmental section, along with relevant environmental data from the company. Your task is to:
        1. Analyze the Text and Data: Carefully review the Environmental section and the E data to understand the company's environmental impacts and the solutions they propose.
        2. Summarize Environmental Impacts: Identify and summarize the key environmental issues associated with the company's operations. This may include factors like carbon emissions, resource consumption, waste generation, biodiversity impact, etc.
        3. Outline Proposed Solutions: Detail the strategies and initiatives the company has proposed or implemented to address these environmental impacts. Highlight any goals, targets, or commitments made.
        4. Include Relevant Metrics: Incorporate significant data points and metrics that support your analysis, such as emission reduction percentages, resource efficiency improvements, or sustainability certifications achieved.
        5. Provide a Comprehensive Summary: Craft a clear, concise, and informative summary that reflects an expert understanding of environmental sustainability. The summary should be suitable for stakeholders interested in the company's environmental performance.
        Please ensure that your analysis is unbiased, fact-based, and emphasizes both the impacts and the effectiveness of the proposed solutions. The data is {e_data}. Please provide a summary of about 200 words accordingly.
    """
    return call_ollama(ENV_PROMPT)

def analyze_social_data(s_data: str) -> str:
    """Analyze the social aspect of the company's ESG data."""
    # prompt = f"Analyze the following social data: {s_data}"
    SOC_PROMPT = f"""
        You are an expert in social sustainability and analysis. You have been provided with an ESG report, focusing on the Social section, along with relevant social data from the company. Your task is to:
        1. Analyze the Text and Data: Carefully review the Social section and the S data to understand the company's social impacts and the solutions they propose.
        2. Summarize Social Impacts: Identify and summarize the key social issues associated with the company's operations. This may include factors like labor practices, employee well-being, diversity and inclusion, community engagement, human rights, etc.
        3. Outline Proposed Solutions: Detail the strategies and initiatives the company has proposed or implemented to address these social impacts. Highlight any goals, targets, or commitments made.
        4. Include Relevant Metrics: Incorporate significant data points and metrics that support your analysis, such as employee turnover rates, diversity statistics, training hours per employee, community investment amounts, etc.
        5. Provide a Comprehensive Summary: Craft a clear, concise, and informative summary that reflects an expert understanding of social sustainability. The summary should be suitable for stakeholders interested in the company's social performance.
        Please ensure that your analysis is unbiased, fact-based, and emphasizes both the impacts and the effectiveness of the proposed solutions. The data is {s_data}. Please provide a summary of about 200 words accordingly.
    """
    return call_ollama(SOC_PROMPT)

def analyze_governance_data(g_data: str) -> str:
    """Analyze the governance aspect of the company's ESG data."""
    # prompt = f"Analyze the following governance data: {g_data}"
    GOV_PROMPT = f"""
        You are an expert in corporate governance and analysis. You have been provided with an ESG report, focusing on the Governance section, along with relevant governance data from the company. Your task is to:
        1. Analyze the Text and Data: Carefully review the Governance section and the G data to understand the company's governance practices and the solutions they propose.
        2. Summarize Governance Issues: Identify and summarize the key governance challenges associated with the company's operations. This may include factors like board diversity, executive compensation, shareholder rights, ethical conduct, transparency, compliance, risk management, etc.
        3. Outline Proposed Solutions: Detail the strategies and initiatives the company has proposed or implemented to address these governance issues. Highlight any goals, targets, or commitments made.
        4. Include Relevant Metrics: Incorporate significant data points and metrics that support your analysis, such as percentages of independent board members, audit results, instances of regulatory compliance, or codes of conduct adherence rates.
        5. Provide a Comprehensive Summary: Craft a clear, concise, and informative summary that reflects an expert understanding of corporate governance. The summary should be suitable for stakeholders interested in the company's governance performance.
        Please ensure that your analysis is unbiased, fact-based, and emphasizes both the issues and the effectiveness of the proposed solutions. The data is {g_data}. Please provide a summary of about 200 words accordingly.
    """
    return call_ollama(GOV_PROMPT)

# Summarizer function to combine all analyses
def summarize_esg(environmental_summary, social_summary, governance_summary):
    # prompt = (
    #     f"Combine the following analyses into a final ESG summary:\n"
    #     f"Environmental: {environmental_summary}\n"
    #     f"Social: {social_summary}\n"
    #     f"Governance: {governance_summary}"
    # )
    SUMMARY_PROMPT = f"""
        You are an expert in financial analysis and ESG report analysis. You have been provided with separate summaries of the company's Environmental, Social, and Governance (ESG) performance, each approximately 200 words. Your task is to:
        1. **Create a Unified Summary:** Combine the three summaries into a cohesive overview that describes and provides an overview of the company's overall ESG performance.
        2. **Keep it Short and Crisp:** Ensure the unified summary is concise, highlighting the most significant points from each section without unnecessary elaboration.
        3. **Assign an ESG Score:** Based on the information in the summaries, provide a score out of 10 for each of the 3 categories and finally a score that represents the company's ESG efforts and commitment. Include a brief justification for the score. Something like: Environmental: 6/10, Social: 7/10, Governance: 8/10, Total: 7/10
        Please make sure the final summary is suitable for stakeholders interested in an overview of the company's ESG performance and reflects your expert analysis. The last line of the provided output should be a score out of 10.
        The provided summaries are:
        - Environmental Summary: {environmental_summary}
        - Social Summary: {social_summary}
        - Governance Summary: {governance_summary}
        Please provide the unified summary and the ESG score accordingly.
    """
    return call_ollama(SUMMARY_PROMPT)

# Main function to handle ESG analysis
def esg_analysis(company_data):
    # Step 1: Analyze individual ESG components
    start = time.time()
    print(1)
    environmental_summary = analyze_environmental_data(company_data["e"])
    print(2)
    e = time.time()
    social_summary = analyze_social_data(company_data["s"])
    s = time.time()
    governance_summary = analyze_governance_data(company_data["g"])
    g = time.time()
    mid = time.time()
    # Step 2: Summarize all analyses
    final_summary = summarize_esg(environmental_summary, social_summary, governance_summary)
    end = time.time()
    e_time = e - start
    s_time = s - start
    g_time = g - start
    mid_time = mid - start
    total_time = end - start
    print(f"Environmental time taken: {e_time:.2f} seconds")
    print(f"Social time taken: {s_time:.2f} seconds")
    print(f"Governance time taken: {g_time:.2f} seconds")
    print(f"Total ESG Agent time taken: {mid_time:.2f} seconds")
    print(f"Total time taken by all 4 agents: {total_time:.2f} seconds")
    # print("Environmental Summary", environmental_summary)
    # print("Social Summary", social_summary)
    # print("Governance Summary", governance_summary)
    # print("Final Summary", final_summary)
    return final_summary

# Example usage
if __name__ == "__main__":
    # esg_data = {
    #     "company_1": {
    #         "e": "The company reduced carbon emissions by 20% and improved renewable energy use.",
    #         "s": "The company achieved a diversity score of 85 and conducted extensive community outreach.",
    #         "g": "The company maintained 75% board independence and adhered to all audit compliance standards.",
    #     }
    # }
    with open('data.json', 'r') as file:
        esg_data = json.load(file)

    for company, data in esg_data.items():
        print(f"ESG Report for {company}:")
        report = esg_analysis(data)
        today_date = datetime.now().strftime("%Y-%m-%d")
        file_name = f"{company}_esg_summary_{today_date}.txt"
        folder_path = f"./reports/multi-agent/{today_date}/{company}"
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "w") as file:
            file.write(report)
        print(f"Report saved to {file_path}")