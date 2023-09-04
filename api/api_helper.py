import guidance
import openai
# connect to a chat model like GPT-4 or Vicuna
openai.api_key = "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"
gpt4 = guidance.llms.OpenAI("gpt-4")
# vicuna = guidance.llms.transformers.Vicuna("your_path/vicuna_13B", device_map="auto")

experts = guidance('''
{{#system~}}
You are a helpful and terse assistant.
{{~/system}}

{{#user~}}
I want a response to the following question:
{{query}}
Name 3 world-class experts (past or present) who would be great at answering this?
Don't answer the question yet.
{{~/user}}

{{#assistant~}}
{{gen 'expert_names' temperature=0 max_tokens=300}}
{{~/assistant}}

{{#user~}}
Great, now please answer the question as if these experts had collaborated in writing a joint anonymous answer.
{{~/user}}

{{#assistant~}}
{{gen 'answer' temperature=0 max_tokens=500}}
{{~/assistant}}
''', llm=gpt4)

print(experts(query='How can I be more productive?'))