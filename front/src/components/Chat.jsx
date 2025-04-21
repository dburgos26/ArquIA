import React, { useState, useRef } from "react";
import MermaidChart from "./MermaidChart";
import { Box, Paper, List, ListItem, TextField, Button, Typography, Dialog, DialogTitle, DialogContent, ListItemText, IconButton, Badge, Chip, Stack } from "@mui/material";
import AttachFileIcon from "@mui/icons-material/AttachFile";
import CloseIcon from "@mui/icons-material/Close";
import ImageIcon from "@mui/icons-material/Image";

import "../styles/chat.css";

export default function Chat() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [openDialog, setOpenDialog] = useState(false);
    const [selectedInternalMessages, setSelectedInternalMessages] = useState([]);
    const [attachedImages, setAttachedImages] = useState([]);
    const fileInputRef = useRef(null);

    const sendMessage = () => {
        if (!input.trim() && attachedImages.length === 0) return;

        setMessages([...messages, {
            sender: "usuario",
            text: input,
            images: attachedImages.map(img => img.preview)
        }]);

        // Create FormData with message and images
        const formData = new FormData();
        formData.append("message", input);
        attachedImages.forEach((img, index) => {
            formData.append(`image${index + 1}`, img.file);
        });

        fetch("http://localhost:8000/message", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                setMessages((prev) => [
                    ...prev,
                    { sender: "respuesta", text: data.endMessage, internal_messages: data.messages, mermaidCode: data.mermaidCode },
                ]);
            })
            .catch((error) => {
                console.error("Error:", error);
            });

        setInput("");
        setAttachedImages([]);
    };

    const handleResponseClick = (internalMessages) => {
        if (internalMessages && internalMessages.length > 0) {
            setSelectedInternalMessages(internalMessages);
            setOpenDialog(true);
        }
    };

    const handleImageUpload = (e) => {
        const files = Array.from(e.target.files);

        // Limit to 2 images total
        if (files.length + attachedImages.length > 2) {
            alert("Solo puedes adjuntar hasta 2 imágenes.");
            return;
        }

        const newImages = files.map(file => ({
            file,
            preview: URL.createObjectURL(file),
            name: file.name
        }));

        setAttachedImages([...attachedImages, ...newImages]);

        // Reset file input
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const removeImage = (indexToRemove) => {
        setAttachedImages(attachedImages.filter((_, index) => index !== indexToRemove));
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
                            <Box sx={{ width: "100%" }}>
                                <Typography variant="body1" sx={{ color: "white", whiteSpace: "pre-wrap"  }}>
                                    {msg.text}
                                </Typography>

                                {/* Display mermaid code */}
                                {msg.mermaidCode && (
                                    <Box sx={{ mt: 2 }}>
                                        <p>Diagrama:</p>
                                        <MermaidChart chart={msg.mermaidCode} />
                                    </Box>
                                )}

                                {/* Display attached images in messages */}
                                {msg.images && msg.images.length > 0 && (
                                    <Box className="image-container">
                                        {msg.images.map((imgSrc, imgIndex) => (
                                            <Box
                                                key={imgIndex}
                                                component="img"
                                                src={imgSrc}
                                                className="message-image"
                                                alt={`Attached image ${imgIndex + 1}`}
                                            />
                                        ))}
                                    </Box>
                                )}
                            </Box>
                        </ListItem>
                    ))}
                </List>
            </Paper>

            {/* Image preview chips */}
            {attachedImages.length > 0 && (
                <Stack
                    direction="row"
                    spacing={1}
                    sx={{
                        mt: 2,  // Add top margin to create space after messages box
                        mb: 2,  // Add bottom margin before input field
                        ml: 1
                    }}
                >
                    {attachedImages.map((img, index) => (
                        <Chip
                            key={index}
                            icon={<ImageIcon />}
                            label={img.name.length > 15 ? img.name.substring(0, 12) + "..." : img.name}
                            onDelete={() => removeImage(index)}
                            sx={{
                                maxWidth: "200px",
                                "& .MuiChip-label": {
                                    whiteSpace: "nowrap",
                                    color: "white"
                                }
                            }}
                        />
                    ))}
                </Stack>
            )}

            {/* Barra de Entrada */}
            <Box className="input-container">
                <TextField
                    className="input-field"
                    fullWidth
                    variant="outlined"
                    placeholder="Escribe un mensaje..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
                    InputProps={{
                        endAdornment: (
                            <IconButton
                                className="attach-button"
                                onClick={() => fileInputRef.current.click()}
                                disabled={attachedImages.length >= 2}
                            >
                                <Badge badgeContent={attachedImages.length} color="primary">
                                    <AttachFileIcon />
                                </Badge>
                            </IconButton>
                        ),
                    }}
                />
                <input
                    type="file"
                    multiple
                    accept="image/*"
                    ref={fileInputRef}
                    onChange={handleImageUpload}
                    style={{ display: "none" }}
                />
                <Button className="send-button" variant="contained" onClick={sendMessage}>
                    Enviar
                </Button>
            </Box>

            <Dialog className="nodes-dialog" open={openDialog} onClose={() => setOpenDialog(false)}>
                <DialogTitle>
                    Mensajes Internos
                    <IconButton
                        aria-label="close"
                        onClick={() => setOpenDialog(false)}
                        sx={{ position: "absolute", right: 8, top: 8 }}
                    >
                        <CloseIcon />
                    </IconButton>
                </DialogTitle>
                <DialogContent>
                    <List>
                        {selectedInternalMessages.map((msg, index) => (
                            <ListItem key={index}>
                                <ListItemText
                                    primary={`${msg.name}: ${msg.content}`}
                                    primaryTypographyProps={{ color: "white" }}
                                />
                            </ListItem>
                        ))}
                    </List>
                </DialogContent>
            </Dialog>
        </Box>
    );
}