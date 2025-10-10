var socket = io('/');
var messagesContainer = document.getElementById('messages');

socket.on('connect', function() {
  console.log('connected to server!');  
});

function sendMessage() {
  var inputField = document.getElementById('message-input');
  if(inputField.value != "") {
    socket.emit('message', {'text': inputField.value});
    inputField.value = "";
  }
}

socket.on('reply', function(data) {
  var li = document.createElement("li");
  li.appendChild(document.createTextNode(data.text));
  messagesContainer.appendChild(li);
});
