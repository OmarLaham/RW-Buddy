from imports import *
# Load .env vars
load_dotenv()	

def client_chat(client, user_input, ass_content=""):
    """Sends a chat message to OpenAI client

    Parameters
    ----------
    client: openai.OpenAI 
        The openai.OpenAI used to chat with OpenAI
    user_input: str
        The msg sent by user
    ass_content: str, optional
        The assisiting content passed to the client to help generate better response

    Returns
    -------
    str
        A string message returned by chat client as response
    """
    
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a daily life coach."},
            {"role": "assistant", "content": ass_content},
            {"role": "user", "content": user_input},
        ],
        temperature=0.3,
        max_tokens=350,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    client_res = res.choices[0].message.content
    return client_res
