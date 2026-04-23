import { createTheme } from '@mui/material/styles';

const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#9c27b0', // Purple
            light: '#d05ce3',
            dark: '#6a0080',
        },
        secondary: {
            main: '#00e5ff', // Cyan
            light: '#6effff',
            dark: '#00b2cc',
        },
        background: {
            default: '#0a0a0f', // Very dark blue/black
            paper: '#13131a',   // Slightly lighter for cards
        },
        text: {
            primary: '#ffffff',
            secondary: '#b3b3b3',
        },
    },
    typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
        h1: {
            fontWeight: 800,
            letterSpacing: '-0.02em',
        },
        h2: {
            fontWeight: 700,
            letterSpacing: '-0.01em',
        },
        button: {
            textTransform: 'none',
            fontWeight: 600,
        },
    },
    components: {
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 8,
                    padding: '10px 24px',
                    transition: 'all 0.3s ease',
                },
                containedPrimary: {
                    background: 'linear-gradient(45deg, #9c27b0 30%, #ff4081 90%)',
                    boxShadow: '0 3px 15px 2px rgba(156, 39, 176, .3)',
                    '&:hover': {
                        background: 'linear-gradient(45deg, #7b1fa2 30%, #f50057 90%)',
                        boxShadow: '0 6px 20px 4px rgba(156, 39, 176, .4)',
                        transform: 'translateY(-2px)',
                    },
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    borderRadius: 16,
                    backgroundImage: 'none', // Remove default standard elevation background in dark mode
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                    boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
                    backdropFilter: 'blur(10px)',
                    backgroundColor: 'rgba(19, 19, 26, 0.7)',
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundImage: 'none',
                },
            },
        },
        MuiTextField: {
            styleOverrides: {
                root: {
                    '& .MuiOutlinedInput-root': {
                        borderRadius: 8,
                        backgroundColor: 'rgba(255, 255, 255, 0.03)',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                            backgroundColor: 'rgba(255, 255, 255, 0.05)',
                        },
                        '&.Mui-focused': {
                            backgroundColor: 'rgba(255, 255, 255, 0.05)',
                            boxShadow: '0 0 0 2px rgba(156, 39, 176, 0.2)',
                        },
                    },
                },
            },
        },
    },
});

export default theme;
