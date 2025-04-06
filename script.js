// Example frontend code to interact with RAG backend
async function askQuestion(question, conversationId = null) {
    const response = await fetch('https://your-backend-url/api/rag-chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: question,
            conversation_id: conversationId
        }),
    });
    
    const data = await response.json();
    
    // Display response
    displayResponse(data.response);
    
    // Display references if needed
    if (data.references && data.references.length > 0) {
        displayReferences(data.references);
    }
    
    return data.conversation_id;

    
}