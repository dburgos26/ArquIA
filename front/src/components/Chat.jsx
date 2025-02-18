import React, { useState } from "react";
import { Box, TextField, Button, List, ListItem, Typography, Paper, Dialog, DialogTitle, DialogContent, ListItemText } from "@mui/material";
import "../styles/chat.css";

export default function Chat() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [openDialog, setOpenDialog] = useState(false);
    const [selectedInternalMessages, setSelectedInternalMessages] = useState([]);

    const sendMessage = () => {
        if (!input.trim()) return;

        setMessages([...messages, { sender: "usuario", text: input }]);

        const formData = new FormData();
        formData.append("message", input);

        fetch("http://localhost:8000/message", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                setMessages((prev) => [
                    ...prev,
                    { sender: "respuesta", text: data.last_message, internal_messages: data.messages },
                ]);
                console.log(messages);
            })
            .catch((error) => {
                console.error("Error:", error);
            });


        setInput("");
    };

    const handleResponseClick = (internalMessages) => {
        if (internalMessages && internalMessages.length > 0) {
            setSelectedInternalMessages(internalMessages);
            setOpenDialog(true);
        }
    };

    return (
        <Box className="chat-container">
            {/* Lista de Mensajes */}
            <Paper className="messages-box">
                <List>
                    {messages.map((msg, index) => (
                        <ListItem
                            key={index}
                            className={msg.sender === "usuario" ? "user-message" : "bot-message"}
                            onClick={() => msg.sender === "respuesta" && handleResponseClick(msg.internal_messages)}
                            style={{ cursor: msg.sender === "respuesta" ? "pointer" : "default" }}
                        >
                            <Typography>{msg.text}</Typography>
                        </ListItem>
                    ))}
                </List>
            </Paper>

            {/* Barra de Entrada */}
            <Box className="input-container">
                <TextField
                    fullWidth
                    variant="outlined"
                    placeholder="Escribe un mensaje..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                />
                <Button variant="contained" color="primary" onClick={sendMessage}>
                    Enviar
                </Button>
            </Box>

            {/* Modal para Mostrar Internal Messages */}
            <Dialog open={openDialog} onClose={() => setOpenDialog(false)}>
                <DialogTitle>Mensajes Internos</DialogTitle>
                <DialogContent>
                    <List>
                        {selectedInternalMessages.map((msg, index) => (
                            <ListItem key={index}>
                                <ListItemText primary={`${msg.name}: ${msg.content}`} />
                            </ListItem>
                        ))}
                    </List>
                </DialogContent>
            </Dialog>
        </Box>
    );
}
