from langchain import OpenAI, LLMChain

# Initialize the language model
llm = OpenAI(api_key="your_openai_api_key")

# Create an answer generation chain
answer_chain = LLMChain(llm=llm, prompt="Answer this question with detail: {input}")

# Create a chain for summarizing the answer
summary_chain = LLMChain(llm=llm, prompt="Summarize this: {input}")

# Main application loop
while True:
    user_input = input("Ask a question about technology (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    
    answer = answer_chain.run(input=user_input)
    summary = summary_chain.run(input=answer)
    
    print("Answer:", answer)
    print("Summary:", summary)