import { createTheme, ThemeProvider } from "@mui/material/styles";

import Header from "./components/Header"; 
import Chat from "./components/Chat";

const theme = createTheme({
  typography: {
    fontFamily: "'Merriweather', serif",
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Header />
      <Chat />
    </ThemeProvider>
  );
}

export default App;