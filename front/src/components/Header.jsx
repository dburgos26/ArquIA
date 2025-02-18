import React from "react";
import { AppBar, Toolbar, Typography, IconButton, Box } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import "../styles/header.css"; 

export default function Header() {
  return (
    <AppBar position="static" className="app-bar">
      <Toolbar sx={{ display: "flex", alignItems: "center" }}>
        <IconButton edge="start" className="menu-button" aria-label="menu">
          <MenuIcon />
        </IconButton>

        <Box sx={{ flexGrow: 1, display: "flex", alignItems: "center", padding: 1 }}>
          <Typography variant="h6" className="title" sx={{ marginRight: 4 }}>
            ArquIA
          </Typography>

          <Typography variant="h6" className="chat-text" sx={{ color: "#d3d3d3" }}>
            Chat
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
}
