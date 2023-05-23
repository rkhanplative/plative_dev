import guidance

# we use StableLM as an open example, but these issues impact all models to varying degrees
guidance.llm = guidance.llms.Transformers("stabilityai/stablelm-base-alpha-3b", device=0)

# we turn token healing off so that guidance acts like a normal prompting library
program = guidance('''The link is <a href="http:{{gen max_tokens=10 token_healing=False}}''')
program()