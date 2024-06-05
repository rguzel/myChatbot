function sendMessage() {
    var userInput = document.getElementById('user-input').value;
    var apiUrl = 'http://127.0.0.1:5000/find_similar'; // Replace with your actual API URL
    fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userInput })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data); // For debugging
        //displayMessage('Bot', data.results.map(result => `${result.title}: ${result.similarity_score}`).join('\n'));
        displayMessage('Bot', `${data.results[0].title}: ${data.results[0].similarity_score}`).join('\n')
    })
    .catch(error => console.error('Error:', error));
}

function displayMessage(sender, message) {
    var conversation = document.getElementById('conversation');
    var newMessage = document.createElement('li');
    newMessage.textContent = sender + ': ' + message;
    conversation.appendChild(newMessage);
}

document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});
