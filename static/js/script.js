function sendMessage() {
    let userInput = document.getElementById("user-input").value;
    let chatBox = document.getElementById("chat-box");

    if (userInput.trim() === "") return; // Ignore empty messages

    // Display user message
    chatBox.innerHTML += `<div class="user-message">You: ${userInput}</div>`;

    // Send message to Flask backend
    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // Display bot response
        chatBox.innerHTML += `<div class="bot-message">Bot: ${data.response}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll
    });

    document.getElementById("user-input").value = ""; // Clear input field
}
