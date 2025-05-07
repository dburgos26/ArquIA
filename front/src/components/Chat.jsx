import React, { useState, useRef } from "react";
import MermaidChart from "./MermaidChart";
import { Box, Paper, List, ListItem, TextField, Button, Typography, Dialog, DialogTitle, DialogContent, ListItemText, IconButton, Badge, Chip, Stack } from "@mui/material";
import AttachFileIcon from "@mui/icons-material/AttachFile";
import CloseIcon from "@mui/icons-material/Close";
import ImageIcon from "@mui/icons-material/Image";
import ThumbUpIcon from "@mui/icons-material/ThumbUp";
import ThumbDownIcon from "@mui/icons-material/ThumbDown";

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
        formData.append("session_id", "2")
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
                    { sender: "respuesta", text: data.endMessage, internal_messages: data.messages, mermaidCode: data.mermaidCode, session_id: data.session_id, message_id: data.message_id},
                ]);
            })
            .catch((error) => {
                console.error("Error:", error);
            });

        setInput("");
        setAttachedImages([]);
    };

    const handleThumbClick = (session_id, message_id, thumbs_up, thumbs_down) => {
        const formdata = new FormData();
        formdata.append("session_id", session_id);
        formdata.append("message_id", message_id);
        formdata.append("thumbs_up", thumbs_up);
        formdata.append("thumbs_down", thumbs_down);
    
        fetch("http://localhost:8000/feedback", {
            method: "POST",
            body: formdata,
        })
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
    
    // Add state for tracking rated messages
    const [ratedMessages, setRatedMessages] = useState(new Set());
    
    // Function to handle rating and prevent double-rating
    const handleRating = (sessionId, messageId, isThumbsUp, msg) => {
        console.log(msg)
        if (ratedMessages.has(`${sessionId}-${messageId}`)) {
            return; // Already rated
        }
        
        // Update state to mark this message as rated
        const newRatedMessages = new Set(ratedMessages);
        newRatedMessages.add(`${sessionId}-${messageId}`);
        setRatedMessages(newRatedMessages);
        
        // Call the existing thumb click handler
        handleThumbClick(sessionId, messageId, isThumbsUp ? 1 : 0, isThumbsUp ? 0 : 1);
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
                                <Typography variant="body1" sx={{ color: "white", whiteSpace: "pre-wrap" }}>
                                    {msg.text}
                                </Typography>
    
                                {/* Display mermaid code */}
                                {msg.mermaidCode && (
                                    <Box sx={{ mt: 2 }}>
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
                                
                                {/* Rating buttons - only show for bot messages and if they have a message_id */}
                                {msg.sender === "respuesta" && (
                                    <Box 
                                        sx={{ 
                                            display: 'flex', 
                                            justifyContent: 'flex-end', 
                                            mt: 1,
                                            opacity: ratedMessages.has(`${msg.session_id}-${msg.message_id}`) ? 0.5 : 1
                                        }}
                                    >
                                        <Typography 
                                            variant="caption" 
                                            sx={{ 
                                                color: 'rgba(255,255,255,0.7)', 
                                                mr: 1, 
                                                alignSelf: 'center' 
                                            }}
                                        >
                                            ¿Fue útil esta respuesta?
                                        </Typography>
                                        <IconButton 
                                            size="small" 
                                            onClick={(e) => {
                                                e.stopPropagation(); // Prevent triggering the ListItem click
                                                handleRating(msg.session_id, msg.message_id, true, msg);
                                            }}
                                            disabled={ratedMessages.has(`${msg.session_id}-${msg.message_id}`)}
                                            sx={{ 
                                                color: 'rgba(255,255,255,0.7)',
                                                '&:hover': { color: '#4caf50' }
                                            }}
                                        >
                                            <ThumbUpIcon fontSize="small" />
                                        </IconButton>
                                        <IconButton 
                                            size="small" 
                                            onClick={(e) => {
                                                e.stopPropagation(); // Prevent triggering the ListItem click
                                                handleRating(msg.session_id, msg.message_id, false, msg);
                                            }}
                                            disabled={ratedMessages.has(`${msg.session_id}-${msg.message_id}`)}
                                            sx={{ 
                                                color: 'rgba(255,255,255,0.7)',
                                                '&:hover': { color: '#f44336' }
                                            }}
                                        >
                                            <ThumbDownIcon fontSize="small" />
                                        </IconButton>
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