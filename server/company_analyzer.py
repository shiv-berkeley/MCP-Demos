from typing import Any
import os
import httpx
import json
import asyncio
from mcp.server.fastmcp import FastMCP

# At the top of company_analyzer.py
import logging
logging.basicConfig(filename='company_analyzer.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(message)s')

# Initialize FastMCP server
mcp = FastMCP("domain-analyzer")

# Constants
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "sonar"
SERPER_API_URL = "https://google.serper.dev/search"
SERPER_API_KEY = "<SERPER_API_KEY>"

async def make_perplexity_request(prompt: str) -> dict[str, Any] | None:
    """Make a request to the Perplexity API with proper error handling."""
    logging.debug("------ ENTERED MAKE PERPLEXITY REQUEST")
    api_key = "<PERPLEXITY_API_KEY>"
    if not api_key:
        return None
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes websites and provides structured information about product features, pricing, and marketing language."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 2000
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                PERPLEXITY_API_URL,
                headers=headers,
                json=payload,
                timeout=60.0
            )
            
            logging.debug(f"------ RESPONSE STATUS: {response.status_code}")
            
            if response.status_code != 200:
                logging.debug(f"------ ERROR RESPONSE: {response.text}")
                return None
                
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.debug(f"------ EXCEPTION: {str(e)}")
            return None

@mcp.tool()
async def analyze_news_and_press(domain: str) -> str:
    """Analyze news outlets and press releases for a domain to identify announcements.
    
    Args:
        domain: The domain name to analyze (e.g., "example.com")
    """
    logging.debug("------ ENTERED ANALYZE NEWS AND PRESS")
    prompt = f"""
    Analyze news outlets and press releases for {domain} and provide a detailed report on:
    
    Objective: News Outlets & Press Releases: Identify announcements on expansions, partnerships, or strategy shifts.
    
    Please include:
    1. Recent company announcements and press releases
    2. Expansion plans or new market entries
    3. Strategic partnerships or collaborations
    4. Major strategy shifts or pivots
    5. Product launches or significant updates
    
    Provide a comprehensive paragraph with specific details found in news articles and press releases.
    """
    
    logging.debug(f"------ PROMPT: {prompt}")
    data = await make_perplexity_request(prompt)
    logging.debug(f"------ DATA: {data}")
    if not data:
        return "Unable to analyze news and press releases. Please check your API key or try again later."
    
    return format_analysis(data)

async def make_serper_request(query: str) -> dict[str, Any] | None:
    """Make a request to the Serper API with proper error handling."""
    logging.debug("------ ENTERED MAKE SERPER REQUEST")
    
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": query,
        "num": 10
    }
    
    logging.debug(f"------ SERPER QUERY: {query}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                SERPER_API_URL,
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            logging.debug(f"------ SERPER RESPONSE STATUS: {response.status_code}")
            
            if response.status_code != 200:
                logging.debug(f"------ SERPER ERROR RESPONSE: {response.text}")
                return None
                
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.debug(f"------ SERPER EXCEPTION: {str(e)}")
            return None

@mcp.tool()
async def analyze_hiring_trends(domain: str) -> str:
    """Analyze LinkedIn profiles and job postings to track hiring trends.
    
    Args:
        domain: The domain name to analyze (e.g., "example.com")
    """
    logging.debug("------ ENTERED ANALYZE HIRING TRENDS")
    
    # Extract company name from domain (simple approach)
    company_name = domain
    
    # First, search for job postings
    job_query = f"{company_name} careers jobs hiring"
    job_data = await make_serper_request(job_query)
    
    # Then search for LinkedIn information
    linkedin_query = f"{company_name} linkedin hiring trends"
    linkedin_data = await make_serper_request(linkedin_query)
    
    # If we have search results, use Perplexity to analyze them
    if job_data or linkedin_data:
        # Prepare search results for analysis
        search_results = []
        
        if job_data and "organic" in job_data:
            for result in job_data["organic"][:5]:
                search_results.append(f"Job Result: {result.get('title', '')} - {result.get('snippet', '')}")
        
        if linkedin_data and "organic" in linkedin_data:
            for result in linkedin_data["organic"][:5]:
                search_results.append(f"LinkedIn Result: {result.get('title', '')} - {result.get('snippet', '')}")
        
        logging.debug(f"------ SEARCH RESULTS: {search_results}")
        
        # Check if we actually have search results to analyze
        if not search_results:
            logging.debug("------ NO SEARCH RESULTS FOUND")
            return "Unable to analyze hiring trends. No relevant job postings or LinkedIn information found."
        
        # Create a prompt for Perplexity to analyze the search results
        search_results_text = "\n".join(search_results)
        prompt = f"""
        Analyze the following search results about {company_name}'s hiring trends and job postings:
        
        {search_results_text}
        
        Objective: LinkedIn or Job Postings: Track hiring trends to predict future initiatives.
        
        Please provide:
        1. Current hiring focus areas and departments
        2. Skills and expertise the company is seeking
        3. Potential new initiatives or projects based on hiring patterns
        4. Geographic expansion indicated by job locations
        5. Overall growth trajectory based on hiring volume and positions
        
        Provide a comprehensive paragraph with specific details found in the search results.
        """
        
        # Use Perplexity to analyze the search results
        analysis_data = await make_perplexity_request(prompt)
        logging.debug(f"------ ANALYSIS DATA: {analysis_data}")
        
        if analysis_data and "choices" in analysis_data and len(analysis_data["choices"]) > 0:
            return format_analysis(analysis_data)
        else:
            logging.debug("------ INVALID ANALYSIS DATA FORMAT")
            return "Unable to analyze hiring trends. Error processing the search results."
    
    logging.debug("------ NO JOB OR LINKEDIN DATA FOUND")
    return "Unable to analyze hiring trends. Insufficient data found for this company."

@mcp.prompt()
def hiring_trends_prompt(domain: str) -> str:
    """Create a prompt to analyze a domain's hiring trends.
    
    Args:
        domain: The domain to analyze
    """
    return f"""
    Please analyze LinkedIn profiles and job postings for {domain} and provide me with information about:
    
    1. Current hiring focus areas
    2. Skills and expertise being sought
    3. Potential new initiatives based on hiring patterns
    4. Geographic expansion indicated by job locations
    5. Overall growth trajectory
    
    I'd like to understand their future initiatives based on hiring trends.
    """

@mcp.prompt()
def news_analysis_prompt(domain: str) -> str:
    """Create a prompt to analyze a domain's news and press releases.
    
    Args:
        domain: The domain to analyze
    """
    return f"""
    Please analyze news outlets and press releases for {domain} and provide me with information about:
    
    1. Recent announcements
    2. Expansion plans
    3. Strategic partnerships
    4. Major strategy shifts
    5. Product launches
    
    I'd like to understand their recent business developments and future direction.
    """


def format_analysis(analysis_data: dict) -> str:
    """Format the analysis data into a readable string."""
    if not analysis_data or "choices" not in analysis_data:
        return "Unable to generate analysis."
        
    return analysis_data["choices"][0]["message"]["content"]

@mcp.tool()
async def analyze_product_features(domain: str) -> str:
    """Analyze a domain to identify product features, pricing, and marketing language.
    
    Args:
        domain: The domain name to analyze (e.g., "example.com")
    """
    logging.debug("------ ENTERED ANALYZE PRODUCT FEATURES")
    prompt = f"""
    Analyze the website at {domain} and provide a detailed report with the following sections:
    
    1. Product Features: Identify the main product features and capabilities offered.
    2. Pricing Structure: Determine the pricing model, tiers, and any free options.
    3. Marketing Language: Analyze the key marketing messages, value propositions, and target audience.
    
    For each section, provide a comprehensive paragraph with specific details found on the website.
    """
    logging.debug(f"------ PROMPT: {prompt}")
    data = await make_perplexity_request(prompt)
    logging.debug(f"------ DATA: {data}")
    if not data:
        return "Unable to analyze the domain. Please check your API key or try again later."
    
    return format_analysis(data)

@mcp.prompt()
def domain_analysis_prompt(domain: str) -> str:
    """Create a prompt to analyze a domain's product features, pricing, and marketing.
    
    Args:
        domain: The domain to analyze
    """
    return f"""
    Please analyze the website {domain} and provide me with information about:
    
    1. Product Features
    2. Pricing Structure
    3. Marketing Language
    
    I'd like to understand what they offer, how they price it, and how they position themselves in the market.
    """

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    # Get domain from user input
    # domain = input("Enter a domain to analyze (e.g. example.com): ")
    
    # # Run analysis
    # result = asyncio.run(analyze_hiring_trends(domain))
    # print("\nAnalysis Results:\n")
    # print(result)
