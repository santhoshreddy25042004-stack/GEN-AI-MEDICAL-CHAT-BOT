#!/usr/bin/env python3

print("Starting Flask app test...")

try:
    print("1. Importing modules...")
    from flask import Flask, render_template, jsonify, request
    print("   Flask imported successfully")
    
    from src.helper import download_hugging_face_embeddings
    print("   Helper imported successfully")
    
    from langchain_pinecone import PineconeVectorStore
    print("   PineconeVectorStore imported successfully")
    
    from langchain_cohere import ChatCohere
    print("   ChatCohere imported successfully")
    
    from langchain.chains import create_retrieval_chain
    print("   Chains imported successfully")
    
    from langchain.chains.combine_documents import create_stuff_documents_chain
    print("   Documents chain imported successfully")
    
    from langchain_core.prompts import ChatPromptTemplate
    print("   Prompts imported successfully")
    
    from dotenv import load_dotenv
    print("   dotenv imported successfully")
    
    from src.prompt import *
    print("   Custom prompts imported successfully")
    
    import os
    print("   os imported successfully")

    print("2. Loading environment variables...")
    load_dotenv()
    
    print("3. Getting API keys...")
    PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
    COHERE_API_KEY=os.environ.get('COHERE_API_KEY')
    print(f"   PINECONE_API_KEY: {'Set' if PINECONE_API_KEY else 'Not set'}")
    print(f"   COHERE_API_KEY: {'Set' if COHERE_API_KEY else 'Not set'}")

    print("4. Setting environment variables...")
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["COHERE_API_KEY"] = COHERE_API_KEY

    print("5. Loading embeddings...")
    embeddings = download_hugging_face_embeddings()
    print("   Embeddings loaded successfully")

    print("6. Connecting to Pinecone...")
    index_name = "medical-bot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print("   Pinecone connection successful")

    print("7. Creating retriever...")
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
    print("   Retriever created successfully")

    print("8. Initializing Cohere model...")
    chatModel = ChatCohere(model="command-r-plus")
    print("   Cohere model initialized successfully")

    print("9. Creating prompt template...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful medical assistant."),
        ("human", "{input}"),
    ])
    print("   Prompt template created successfully")

    print("10. Creating chains...")
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("   RAG chain created successfully")

    print("11. Creating Flask app...")
    app = Flask(__name__)
    print("   Flask app created successfully")

    print("✅ All components initialized successfully!")
    print("Starting Flask server...")
    
    @app.route("/")
    def index():
        return render_template('chat.html')

    @app.route("/get", methods=["GET", "POST"])
    def chat():
        msg = request.form["msg"]
        input = msg
        print(f"Question: {input}")
        response = rag_chain.invoke({"input": msg})
        print(f"Response: {response['answer']}")
        return str(response["answer"])

    app.run(host="0.0.0.0", port=8080, debug=False)

except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()