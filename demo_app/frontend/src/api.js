export const generateMusic = async (params) => {
    try {
        const response = await fetch('http://localhost:5158/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Extract custom headers
        const metrics = {
            generationTime: response.headers.get('X-Generation-Time'),
            tokensPerSecond: response.headers.get('X-Tokens-Per-Second'),
            actualTokens: response.headers.get('X-Actual-Tokens'),
            numBars: response.headers.get('X-Num-Bars'),
            grammarErrorRate: response.headers.get('X-Grammar-Error-Rate'),
            targetReached: response.headers.get('X-Target-Reached') === 'True' || response.headers.get('X-Target-Reached') === 'true',
        };

        const blob = await response.blob();
        return { blob, metrics };
    } catch (error) {
        console.error('Error generating music:', error);
        throw error;
    }
};
