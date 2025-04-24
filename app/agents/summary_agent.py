from PyPDF2 import PdfReader
import re
import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def extract_text_from_pdf(filepath):
    """Extract text from PDF with improved formatting preservation"""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            # Clean up some common PDF extraction issues
            page_text = re.sub(r'(\w)-\s*\n(\w)', r'\1\2', page_text)  # Fix hyphenated words
            text += page_text + "\n"
    return text

def extract_title(text):
    """Extract the paper title using multiple patterns"""
    # Try to find title from common patterns
    patterns = [
        # Title at the beginning followed by authors or abstract
        r"(?i)^([^\n]{10,200})\n(?:.*?\n){1,5}(?:abstract|introduction|authors?|\d{4})",
        # Title with specific formatting
        r"(?i)(?:^|\n)([A-Z][^\n]{10,200})\n(?:.*?\n){1,3}(?:\w+@|\w+\s+university|abstract)",
        # Fallback pattern
        r"(?i)^([^\n]{10,200})\n"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            title = matches[0].strip()
            # Clean up any reference markers or DOIs
            title = re.sub(r'\[\d+\]|\[\d+,\d+\]|https?://\S+|doi:', '', title)
            return title
    
    return "Title not found"

def extract_authors(text):
    """Extract the author information"""
    # Look for author section before abstract
    patterns = [
        r"(?i)(?:^|\n)(?:.*?\n){1,5}((?:[A-Z][^\n]*?){1,100})\n+(?:abstract|keywords)",
        r"(?i)(?:^|\n)((?:.*?(?:university|institute|college|school|department|lab).*?\n){1,5})(?:abstract|keywords)",
        r"(?i)(?:^|\n)(?:.*?\n){1,3}((?:[A-Za-z\s.,]+\n){1,3})(?:abstract|keywords)"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            authors = matches[0].strip()
            # Clean up emails and other non-author information
            authors = re.sub(r'<.*?>|\(.*?\)|{.*?}|\[.*?\]|https?://\S+|@\S+', '', authors)
            return authors
    
    return "Authors not found"

def extract_section(text, section_names, next_section_names=None):
    """
    Extract a section from text based on its name and potential next sections
    
    Args:
        text: The full text of the paper
        section_names: List of possible names for the target section
        next_section_names: List of possible names for sections that might follow
    """
    if next_section_names is None:
        next_section_names = ["References", "Conclusion", "Discussion", "Results", 
                              "Acknowledgements", "Appendix", "Bibliography"]
    
    # Build regex pattern for section headers
    section_pattern = '|'.join([re.escape(name) for name in section_names])
    
    # Patterns for section headers with different formats
    patterns = [
        # Section with number prefix (e.g., "1. Introduction" or "1 Introduction")
        rf"(?i)(?:^|\n)(?:\d+\.?\s+)?({section_pattern})[\s\n]+([^\n].*?)(?=(?:^|\n)(?:\d+\.?\s+)?(?:{next_section_pattern})[\s\n]+|$)",
        # Section with uppercase title
        rf"(?i)(?:^|\n)({section_pattern.upper()})[\s\n]+([^\n].*?)(?=(?:^|\n)(?:{next_section_pattern})[\s\n]+|$)",
        # Simple section start
        rf"(?i)(?:^|\n)({section_pattern})[\s\n]+([^\n].*?)(?=(?:^|\n)(?:{next_section_pattern})[\s\n]+|$)",
    ]
    
    # Add the next section names to the pattern
    next_section_pattern = '|'.join([re.escape(name) for name in next_section_names + section_names])
    
    # Try each pattern
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        if matches:
            section_content = matches.group(2).strip()
            # Limit to reasonable length (avoid capturing too much or entire document)
            if len(section_content) > 20000:  # Limit to approximately 5-6 pages of content
                section_content = section_content[:20000] + "... (section truncated due to length)"
            return section_content
    
    # Fallback - try to find any paragraph containing keywords from section names
    relevant_keywords = set(word.lower() for name in section_names for word in name.split())
    paragraphs = text.split('\n\n')
    relevant_paragraphs = []
    
    for para in paragraphs:
        para_words = set(para.lower().split())
        if any(keyword in para_words for keyword in relevant_keywords):
            relevant_paragraphs.append(para)
    
    if relevant_paragraphs:
        # Return the first 3 relevant paragraphs or fewer if not enough
        return '\n\n'.join(relevant_paragraphs[:3])
    
    return f"Section not found: {', '.join(section_names)}"

def extract_abstract(text):
    """Extract the abstract section"""
    patterns = [
        r"(?i)abstract[\s\n:]+(.*?)(?=(?:\n\n|\n\d|\nIntroduction|\nI\. Introduction))",
        r"(?i)abstract[\s\n:]+(.*?)(?=\n\n)",
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            abstract = matches.group(1).strip()
            # Clean up the abstract text
            abstract = re.sub(r'\n+', ' ', abstract)
            abstract = re.sub(r'\s+', ' ', abstract)
            if 50 <= len(abstract) <= 2000:  # Reasonable abstract length
                return abstract
    
    # Try a broader approach
    if "abstract" in text.lower():
        abstract_start = text.lower().find("abstract")
        excerpt = text[abstract_start:abstract_start+1500]  # Get a chunk after "abstract"
        paragraphs = excerpt.split('\n\n')
        if len(paragraphs) >= 2:
            return paragraphs[1].strip()
    
    return "Abstract not found"

def extract_keywords(text):
    """Extract paper keywords if present"""
    patterns = [
        r"(?i)(?:key\s*words|keywords)[\s\n:]+([^\n]+(?:\n[^\n]+){0,2}?)(?=\n\n|\n\d|\nIntroduction)",
        r"(?i)(?:key\s*words|keywords)[\s\n:]+([^\n]{10,500})"
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            keywords = matches.group(1).strip()
            # Clean up the keywords
            keywords = re.sub(r'\n+', ' ', keywords)
            return keywords
    
    return "Keywords not found"

def extract_topic_keywords(text, num_keywords=10):
    """Extract the most important keywords from the text using frequency analysis"""
    # Clean and tokenize the text
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = clean_text.split()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count word frequencies
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top keywords
    keywords = [word for word, freq in sorted_words[:num_keywords]]
    return keywords

def generate_structured_summary(filepath):
    """Generate a structured summary of the research paper"""
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    
    try:
        text = extract_text_from_pdf(filepath)
        
        # Extract basic paper information
        title = extract_title(text)
        authors = extract_authors(text)
        abstract = extract_abstract(text)
        keywords = extract_keywords(text)
        
        # Extract key sections
        introduction = extract_section(text, 
                                      ["Introduction", "1. Introduction", "I. Introduction", "Background"],
                                      ["Method", "Methodology", "Data", "Approach", "2.", "II."])
        
        methodology = extract_section(text, 
                                     ["Method", "Methodology", "Approach", "Proposed Method", 
                                      "2. Method", "III. Methodology", "Our Approach"],
                                     ["Experimental", "Evaluation", "Results", "3.", "IV."])
        
        dataset = extract_section(text, 
                                 ["Dataset", "Data", "Data Collection", "Experimental Setup", 
                                  "3. Dataset", "Data Description"],
                                 ["Results", "Evaluation", "4.", "V."])
        
        results = extract_section(text, 
                                ["Results", "Evaluation", "Experiments", "4. Results", 
                                 "Experimental Results", "Performance"],
                                ["Discussion", "Conclusion", "5.", "VI."])
        
        discussion = extract_section(text, 
                                   ["Discussion", "Analysis", "5. Discussion"],
                                   ["Conclusion", "Future Work", "6.", "VII."])
        
        conclusion = extract_section(text, 
                                   ["Conclusion", "Conclusions", "6. Conclusion", "Summary", 
                                    "Final Remarks"],
                                   ["References", "Acknowledgement", "Appendix"])
        
        # Extract topic keywords based on content
        topic_keywords = extract_topic_keywords(text)
        
        summary = {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "keywords": keywords,
            "topic_keywords": topic_keywords,
            "introduction": introduction,
            "methodology": methodology,
            "dataset": dataset,
            "results": results,
            "discussion": discussion,
            "conclusion": conclusion
        }
        
        return summary
    
    except Exception as e:
        return {"error": f"Error processing PDF: {str(e)}"}

# Example usage
if __name__ == "__main__":
    sample_pdf = "path_to_your_paper.pdf"
    summary = generate_structured_summary(sample_pdf)
    
    # Print the summary
    for key, value in summary.items():
        print(f"\n{'='*20} {key.upper()} {'='*20}")
        print(value[:500] + "..." if len(str(value)) > 500 else value)