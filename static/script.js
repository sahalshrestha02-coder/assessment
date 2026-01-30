const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatHistory = document.getElementById('chat-history');
const typingIndicator = document.getElementById('typing');

function appendMessage(role, text, category = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    let content = text;

    const metaDiv = document.createElement('div');
    metaDiv.className = 'msg-meta';

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    metaDiv.innerText = time;

    if (category) {
        const tag = document.createElement('span');
        tag.className = 'category-tag';
        tag.innerText = category;
        metaDiv.prepend(tag);
    }

    msgDiv.innerText = content;
    msgDiv.appendChild(metaDiv);

    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const question = userInput.value.trim();
    if (!question) return;

    // Clear input
    userInput.value = '';

    // User Message
    appendMessage('user', question);

    // Show typing
    typingIndicator.style.display = 'flex';
    chatHistory.scrollTop = chatHistory.scrollHeight;

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question }),
        });

        if (!response.ok) throw new Error('API Error');

        const data = await response.json();

        // Hide typing
        typingIndicator.style.display = 'none';

        // Bot Message
        appendMessage('bot', data.answer, data.category);

    } catch (error) {
        typingIndicator.style.display = 'none';
        appendMessage('bot', 'Sorry, I encountered an error. Please try again later.');
        console.error(error);
    }
});
