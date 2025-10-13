from llama_cpp import Llama
from format_prompt import generate_text

LLM_file = "gpt-oss-20b-F16.gguf"

llm = Llama(model_path = LLM_file,  n_ctx=0,  embedding = False)

prompt = generate_text()

output = llm(
    prompt,
    max_tokens=512,  
    #echo = True,
)

print("--------------------------------")
print(output["choices"][0]["text"])
print("--------------------------------")