from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort
)

def generate_text():
 
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    
    system_message = (
        SystemContent.new()
            .with_reasoning_effort(ReasoningEffort.MEDIUM)
            .with_conversation_start_date("2025-06-28")
    )
    
    developer_message = (
        DeveloperContent.new()
            .with_instructions("Always respond politely.")
            
    )
    
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, system_message),
            Message.from_role_and_content(Role.DEVELOPER, developer_message),
            Message.from_role_and_content(Role.USER, "What is the weather in Tokyo?"),
        ]
    )
    
    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    text = encoding.decode_utf8(tokens)
    return text

if __name__ == "__main__":
    print(generate_text())