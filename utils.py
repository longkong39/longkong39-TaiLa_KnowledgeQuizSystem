import yaml
from pathlib import Path
from datetime import datetime
from pathlib import Path
from typing import Union

def make_prompt(text, info):
    if info:
        prompt = f"问题: {text}\n检索到的知识:{info}\n综合提示:基于以上提供的知识，请综合回答问题{text}。如果有必要，可以引用特定的知识片段来支持你的答案，但请确保答案直接、简洁，并且易于理解。"
    else:
        prompt = text
    return prompt

def read_yaml(yaml_path: Union[str, Path]):
    with open(str(yaml_path), "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data

def get_timestamp():
    return datetime.strftime(datetime.now(), "%Y-%m-%d")

def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)