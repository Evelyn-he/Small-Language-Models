# README

## Dependencies to run the SLM

We are using Ollama runtime to host our SLM. To get Ollama working, do the following:
1. Download Ollama through https://ollama.com/download
2. Download the SLM you want to use throut the command line: `ollama run phi3:3.8b`

## Dependencies to run the LLM

To make API calls to ChatGPT, do the following:
1. Go to `https://platform.openai.com/api-keys`
2. Copy the API key and set it into your environment: `$env:OPENAI_API_KEY="<key>"`


## To Run the main loop

1. Make sure correct requirements are installed through `pip install -r requirements.txt`
2. Launch ollama: `ollama list`.
3. In the root folder, run `python cli.py` to spin up the program.