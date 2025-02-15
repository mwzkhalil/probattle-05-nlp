from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import requests

class CustomLLM(LLM):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        url = "https://57a2-34-126-116-155.ngrok-free.app/generate"

        data = {
            "inputs": prompt,
            "parameters": {
                "max_tokens": 50,
                "temperature": 0.7
            }
        }
        response = requests.post(url, json=data)
        print("Status Code", response.status_code)
        
        if response.status_code == 200:
            response_json = response.json()
            print(response_json) 
            if 'choices' in response_json and len(response_json['choices']) > 0:
                generated_text = response_json['choices'][0]['message']['content']
                if generated_text:
                    security_guard_response = generated_text.split("Security Guard:")[-1].strip()
                    
                    return security_guard_response

        return ""  # Return an empty string if the response is not valid

    @property
    def _llm_type(self) -> str:
        return "microsoft/Phi-3-mini-4k-instruct-gguf"