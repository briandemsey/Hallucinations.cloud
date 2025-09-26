# utils/file_processor.py
"""
File Processing Utilities for Hallucinations.cloud
Handles file uploads, content extraction, and prompt enhancement
"""

import json
import pandas as pd
from typing import Dict, Tuple, Any

class FileProcessor:
    """Handles file processing and content extraction"""
    
    def __init__(self):
        self.supported_types = {
            'text/plain': self._process_text_file,
            'application/json': self._process_json_file,
            'text/csv': self._process_csv_file,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self._process_excel_file,
            'application/vnd.ms-excel': self._process_excel_file,
            'text/markdown': self._process_markdown_file
        }
        
        self.max_content_length = 4000  # Truncate content to prevent token overflow
    
    def process_uploaded_file(self, uploaded_file) -> Tuple[str, Dict[str, Any]]:
        """Process uploaded file and extract content"""
        
        file_info = {
            "name": uploaded_file.name,
            "size": uploaded_file.size,
            "type": uploaded_file.type
        }
        
        try:
            # Determine processing method based on file type and extension
            processor = self._get_processor(uploaded_file)
            file_content = processor(uploaded_file)
            
            # Truncate content if too long
            if len(file_content) > self.max_content_length:
                file_content = file_content[:self.max_content_length] + "..."
                file_info["truncated"] = True
            else:
                file_info["truncated"] = False
            
            file_info["processed_successfully"] = True
            
        except Exception as e:
            file_content = f"[Error processing file: {str(e)}]"
            file_info["processed_successfully"] = False
            file_info["error"] = str(e)
        
        return file_content, file_info
    
    def _get_processor(self, uploaded_file):
        """Get the appropriate processor for the file"""
        
        # Check by MIME type first
        if uploaded_file.type in self.supported_types:
            return self.supported_types[uploaded_file.type]
        
        # Check by file extension
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.txt'):
            return self._process_text_file
        elif file_name.endswith('.json'):
            return self._process_json_file
        elif file_name.endswith('.csv'):
            return self._process_csv_file
        elif file_name.endswith(('.xlsx', '.xls')):
            return self._process_excel_file
        elif file_name.endswith('.md'):
            return self._process_markdown_file
        else:
            return self._process_generic_file
    
    def _process_text_file(self, uploaded_file) -> str:
        """Process plain text files"""
        try:
            content = uploaded_file.read().decode('utf-8')
            return f"Text File Content:\n\n{content}"
        except UnicodeDecodeError:
            # Try different encodings
            uploaded_file.seek(0)
            try:
                content = uploaded_file.read().decode('latin1')
                return f"Text File Content (Latin1 encoding):\n\n{content}"
            except Exception:
                return "[Error: Unable to decode text file]"
    
    def _process_json_file(self, uploaded_file) -> str:
        """Process JSON files"""
        try:
            json_data = json.load(uploaded_file)
            formatted_json = json.dumps(json_data, indent=2)
            
            # Provide structure summary for large JSON files
            if len(formatted_json) > 2000:
                structure_info = self._analyze_json_structure(json_data)
                return f"JSON File Structure Analysis:\n{structure_info}\n\nFirst 1000 characters:\n{formatted_json[:1000]}..."
            else:
                return f"JSON File Content:\n\n{formatted_json}"
                
        except json.JSONDecodeError as e:
            return f"[Error: Invalid JSON file - {str(e)}]"
    
    def _process_csv_file(self, uploaded_file) -> str:
        """Process CSV files"""
        try:
            df = pd.read_csv(uploaded_file)
            
            # Create comprehensive summary
            summary = f"CSV Data Analysis:\n"
            summary += f"Rows: {len(df)}\n"
            summary += f"Columns: {len(df.columns)}\n"
            summary += f"Column Names: {list(df.columns)}\n\n"
            
            # Data types
            summary += f"Data Types:\n{df.dtypes.to_string()}\n\n"
            
            # Sample data
            summary += f"First 5 rows:\n{df.head().to_string()}\n\n"
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary += f"Numeric Column Statistics:\n{df[numeric_cols].describe().to_string()}\n\n"
            
            # Missing values
            missing = df.isnull().sum()
            if missing.any():
                summary += f"Missing Values:\n{missing[missing > 0].to_string()}\n"
            
            return summary
            
        except Exception as e:
            return f"[Error processing CSV: {str(e)}]"
    
    def _process_excel_file(self, uploaded_file) -> str:
        """Process Excel files"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            summary = f"Excel File Analysis:\n"
            summary += f"Number of sheets: {len(sheet_names)}\n"
            summary += f"Sheet names: {sheet_names}\n\n"
            
            # Process first sheet in detail
            df = pd.read_excel(uploaded_file, sheet_name=0)
            summary += f"First Sheet ('{sheet_names[0]}') Analysis:\n"
            summary += f"Rows: {len(df)}\n"
            summary += f"Columns: {len(df.columns)}\n"
            summary += f"Column Names: {list(df.columns)}\n\n"
            
            # Sample data from first sheet
            summary += f"First 3 rows:\n{df.head(3).to_string()}\n\n"
            
            # If multiple sheets, provide basic info about others
            if len(sheet_names) > 1:
                summary += "Other Sheets Summary:\n"
                for sheet_name in sheet_names[1:]:
                    try:
                        sheet_df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                        summary += f"- {sheet_name}: {len(sheet_df)} rows, {len(sheet_df.columns)} columns\n"
                    except Exception:
                        summary += f"- {sheet_name}: [Could not read sheet]\n"
            
            return summary
            
        except Exception as e:
            return f"[Error processing Excel file: {str(e)}]"
    
    def _process_markdown_file(self, uploaded_file) -> str:
        """Process Markdown files"""
        try:
            content = uploaded_file.read().decode('utf-8')
            
            # Analyze markdown structure
            lines = content.split('\n')
            headers = [line for line in lines if line.startswith('#')]
            
            summary = f"Markdown File Analysis:\n"
            summary += f"Total lines: {len(lines)}\n"
            summary += f"Headers found: {len(headers)}\n\n"
            
            if headers:
                summary += "Document Structure:\n"
                for header in headers[:10]:  # Show first 10 headers
                    summary += f"{header}\n"
                if len(headers) > 10:
                    summary += f"... and {len(headers) - 10} more headers\n"
                summary += "\n"
            
            summary += f"Content:\n\n{content}"
            return summary
            
        except UnicodeDecodeError:
            return "[Error: Unable to decode markdown file]"
    
    def _process_generic_file(self, uploaded_file) -> str:
        """Process unsupported file types"""
        return f"[{uploaded_file.type} file detected - limited processing available]\n\nFile: {uploaded_file.name}\nSize: {uploaded_file.size} bytes\nType: {uploaded_file.type}"
    
    def _analyze_json_structure(self, json_data, max_depth=3) -> str:
        """Analyze JSON structure for large files"""
        
        def analyze_level(data, depth=0):
            if depth > max_depth:
                return "..."
            
            if isinstance(data, dict):
                if len(data) == 0:
                    return "{}"
                items = []
                for key, value in list(data.items())[:5]:  # Show first 5 keys
                    value_type = type(value).__name__
                    if isinstance(value, (dict, list)):
                        value_desc = analyze_level(value, depth + 1)
                    else:
                        value_desc = f"{value_type}"
                    items.append(f'"{key}": {value_desc}')
                
                if len(data) > 5:
                    items.append(f"... and {len(data) - 5} more keys")
                
                return "{\n" + ",\n".join(f"  {item}" for item in items) + "\n}"
            
            elif isinstance(data, list):
                if len(data) == 0:
                    return "[]"
                elif len(data) == 1:
                    return f"[{analyze_level(data[0], depth + 1)}]"
                else:
                    first_item = analyze_level(data[0], depth + 1)
                    return f"[{first_item}, ... {len(data)} items total]"
            
            else:
                return type(data).__name__
        
        return analyze_level(json_data)
    
    def create_file_enhanced_prompt(self, user_query: str, file_content: str, file_name: str, processing_mode: str = "analyze") -> str:
        """Create enhanced prompt with file context"""
        
        if not file_content or file_content.startswith("[Error"):
            return user_query
        
        mode_instructions = {
            "analyze": "Please analyze this file content in detail and answer the user's question based on the data.",
            "summarize": "Please provide a concise summary of this file and then address the user's question.",
            "extract": "Please extract the key points and insights from this file that relate to the user's question.",
            "question": "Please answer the user's question using the information from this attached file."
        }
        
        enhanced_prompt = f"""I have attached a file named "{file_name}" for context. Here is the file content:

--- FILE CONTENT START ---
{file_content}
--- FILE CONTENT END ---

{mode_instructions.get(processing_mode, "Please analyze this file content and answer the user's question.")}

User Question: {user_query}

Please provide a comprehensive response that takes the attached file data into account. Be specific and reference the file data where relevant.
"""
        
        return enhanced_prompt
    
    def get_file_summary(self, file_info: Dict[str, Any]) -> str:
        """Get a brief summary of the processed file"""
        if not file_info:
            return "No file attached"
        
        summary = f"📎 {file_info['name']} ({file_info['size'] / 1024:.1f} KB)"
        
        if file_info.get("truncated", False):
            summary += " [Truncated]"
        
        if not file_info.get("processed_successfully", True):
            summary += " [Processing Error]"
        
        return summary
