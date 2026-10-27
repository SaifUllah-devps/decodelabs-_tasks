
print("Chatbot: Hello! Welcome to my chatbot.")
print("Chatbot: Type 'bye' or 'exit' to end the conversation.")

while True:
    user_input = input("You: ").lower().strip()

    if user_input in ["hello", "hi", "hey"]:
        print("Chatbot: Hello! How can I help you?")

    elif user_input in ["how are you", "how are you?"]:
        print("Chatbot: I'm doing great! Thanks for asking.")

    elif user_input in ["what is your name", "what's your name"]:
        print("Chatbot: I'm a simple rule-based chatbot.")

    elif user_input in ["what is ai", "what is ai?"]:
        print("Chatbot: AI stands for Artificial Intelligence.")

    elif user_input in ["help", "what can you do"]:
        print("Chatbot: I can greet you, answer basic questions, and chat with you.")

    elif user_input in ["bye", "goodbye", "exit", "quit"]:
        print("Chatbot: Goodbye! Have a great day.")
        break

    else:
        print("Chatbot: Sorry, I don't understand that yet.")
