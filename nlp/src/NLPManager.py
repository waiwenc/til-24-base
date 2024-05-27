import json
import re
from typing import Dict
from word2number import w2n

class NLPManager:
    def __init__(self):
        self.number_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3",
            "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "niner": "9"
        }
        self.target_keywords = [
            "engage", "engaging", "engaged",
            "target", "targeting", "targeted",
            "intercept", "intercepting", "intercepted",
            "neutralize", "neutralizing", "neutralized",
            "suspicious", "incoming", "track", "tracking", "tracked",
            "against", "hostile"
        ]


    def convert_to_number(self, text: str) -> str:
        parts = text.split()
        number_str = ''.join([self.number_map.get(part, '') for part in parts])
        return f"{int(number_str):03d}" if number_str.isdigit() else "000"

    def qa(self, transcript: str) -> Dict[str, str]:
        # target_pattern = r"target(?: is| the| at| located at)?(?: [^,]+)? ([\w\s,]+)"
        target_pattern = r"(?:{})\s+([a-zA-Z\s,]+)".format('|'.join(self.target_keywords))
        heading_pattern = r"heading (?:is |to |at |on |towards )?([\w\s-]+)"
        tool_pattern = r"(?:deploy(?:ing|ed)?|engage(?:s|d)? with|prepare(?:s|d)? to deploy|fire(?:s|d)? at|use(?:s|d)?|intercept(?:s|ed)?|activate(?:s|d)?|utilize(?:s|d)?|launch(?:es|ed)?|focus(?:es|ed)?|aim(?:s)?|release(?:s|d)?) ([\w\s-]+)"

        target_match = re.search(target_pattern, transcript, re.IGNORECASE)
        heading_match = re.search(heading_pattern, transcript, re.IGNORECASE)
        tool_match = re.search(tool_pattern, transcript, re.IGNORECASE)

        target = target_match.group(1).strip() if target_match else "unknown"
        heading = self.convert_to_number(heading_match.group(1)) if heading_match else "000"
        tool = tool_match.group(1).strip() if tool_match else "unknown"

        return {"key": entry['key'], "target": target, "heading": heading, "tool": tool}

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file]

def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    transcripts_path = 'transcripts.jsonl'
    output_path = 'nlp.jsonl'
    manager = NLPManager()

    data = load_data(transcripts_path)
    extracted_data = []

    for entry in data:
        extracted_info = manager.qa(entry['transcript'])
        extracted_data.append(extracted_info)

    save_data(extracted_data, output_path)
    print("Data extraction complete and saved to:", output_path)