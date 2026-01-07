# README

## Dependencies to run the SLM

We are using Ollama runtime to host our SLM. To get Ollama working, do the following:
1. Download Ollama through https://ollama.com/download
2. Download the SLM you want to use throut the command line: `ollama run phi3:3.8b`

## Dependencies to run the LLM

To make API calls to ChatGPT, do the following:
1. Go to `https://platform.openai.com/api-keys`
2. Copy the API key and set it into your environment: `$env:OPENAI_API_KEY="<key>"`

## Dependencies for NER filter function:
Currently the second layer filter function uses spacy, to download the required files for it to run please run
`python -m spacy download en_core_web_sm`

## To Run the main loop

1. Make sure correct requirements are installed through `pip install -r requirements.txt`
2. Launch ollama: `ollama list`.
3. In the root folder, run `python cli.py` to spin up the program.

## To run the server-client api on a single laptop

1. On one terminal, run `python server.py [-p <port>]`, you should see `[SERVER] Listening on <ip>:<port>` message from the output. 
    (`-p` is an optional flag. The port is set to 5001 by default)
2. On another terminal, run `python client.py`, and enter the ip and port you got previously.

## To run the server-client api on different laptops

1. On the laptop that runs the server:
    1. install ngrok
    2. run `python server.py [-p <port>]`
    3. on a new terminal, run `ngrok tcp <port>`. You will see an address in the format of `<digit>.tcp.ngrok.io:<new_port>`
2. On the laptop(s) that run the client(s):
    1. run `python client.py`
    2. Enter `<digit>.tcp.ngrok.io` for ip and enter `<new_port>` for port
