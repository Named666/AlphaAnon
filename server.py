from vllm import LLM, SamplingParams

# Initialize the model from the GRPO_4chan folder
llm = LLM(model="GRPO_4chan")

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)

# Example prompt
prompts = ["Hello! How can I assist you today?"]

# Generate output
outputs = llm.generate(prompts, sampling_params)

# Print the result
for output in outputs:
    print(output.outputs[0].text)