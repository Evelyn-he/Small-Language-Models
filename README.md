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

## Dependencies to connect to MongoDB:
1. Sign up for a MongoDB account here: `https://www.mongodb.com/products/platform/atlas-database`
2. Once you are signed in, send the email address you used to sign in to Evelyn, to be invited to the project.
3. In the project overview, click on the `Connect` button, then click on `Compass`. Copy the connection string into your `.env` with variable name `MONGO_URI`.
4. (Optional but very helpful): Download MongoDB compass at `https://www.mongodb.com/products/tools/compass`. This is a helpful GUI tool that lets you see the data easily.

## To Run the main loop

1. Make sure correct requirements are installed through `pip install -r requirements.txt`
2. Launch ollama: `ollama list`.
3. In the root folder, run `python cli.py` to spin up the program.

## To run the Chatbot UI

1. Install Node.js from `https://nodejs.org/en/download`
2. Go to the UI project folder and install dependencies
    ```
    cd chatbot-ui
    npm install
    ```
3. Go back to the root folder. Start the backend. Wait untill you see `Running on http://127.0.0.1:5001`
    ```
    python api_server.py
    ```
4. On another terminal, go to the UI project folder and start the frontend. You will see something like `Local:   http://localhost:5173/` on your terminal. Copy and paste the url to a browser and you can play with the UI now.
    ```
    cd chatbot-ui
    npm run dev
    ```