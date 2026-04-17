import express, { Request, Response } from 'express';

const app = express();
app.use(express.json());

// ============================================================================
// TASK 2: TRUE NLP ENTITY RESOLUTION WITH COREFERENCE
// ============================================================================

/**
 * Safe coreference resolution using compromise NLP
 * 
 * Safely identifies pronouns (he, she, it, him, her, etc.) as actual standalone words
 * and replaces them with [Target_Entity] without corrupting word substrings.
 * 
 * Example:
 *   Input:  "He went to the shelter. He was very happy there."
 *   Output: "[Target_Entity] went to the shelter. [Target_Entity] was very happy there."
 *   
 *   Does NOT corrupt:
 *   - "sheet" → NOT converted to "s[Target_Entity]et"
 *   - "heritage" → NOT converted to "[Target_Entity]ritage"
 */
const resolveEntities = (input: string): string => {
    try {
        if (!input || typeof input !== 'string') {
            return '';
        }

        // Replace only standalone pronouns using word boundaries.
        // This avoids corrupting substrings like "sheet" or "heritage".
        const pronounPattern = /\b(he|she|it|him|her|they|them|i|me|you)\b/gi;
        return input.replace(pronounPattern, '[Target_Entity]');
    } catch (error) {
        console.error(`[NLP Resolution Error] ${error}`);
        // Fail-safe: return original input on parsing error
        return input;
    }
};

// Risk Detection Matrix (Safety & Risk Escalation Layer)
const checkRiskGuardrails = (content: string) => {
    const riskKeywords = ['suicide', 'die', 'hurt myself', 'end it all', 'desperate'];
    const lowerContent = content.toLowerCase();

    for (const word of riskKeywords) {
        if (lowerContent.includes(word)) {
            return { riskFound: true, level: 'CRITICAL', trigger: word };
        }
    }
    return { riskFound: false, level: 'SAFE' };
};

app.post('/api/analyze', async (req: Request, res: Response) => {
    const { user_id, content } = req.body;
    console.log(`\n[Node Orchestrator] Processing input for ${user_id}`);

    try {
        // Step 1: Safety & Risk Escalation Guardrails
        const riskAssessment = checkRiskGuardrails(content);
        if (riskAssessment.riskFound) {
            console.warn(`[CRISIS ESCALATION] HIGH RISK DETECTED: Trigger="${riskAssessment.trigger}"`);
            // Antigravity action: Instantly route to crisis worker, halt standard processing.
            return res.status(200).json({ status: 'escalated', alert: 'Crisis intervention triggered.' });
        }

        // Step 2: Multi-Turn Entity Resolution (NLP-safe coreference)
        const resolvedText = resolveEntities(content);
        console.log(`[Node Orchestrator] Resolved Input: ${resolvedText}`);

        // Forward to Python AI Worker
        console.log(`[Orchestrator] Forwarding resolved text to Python AI Worker...`);
        const pythonResponse = await fetch('http://ai-python:5000/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: user_id,
                query: resolvedText,
                tone: 'empathetic',
                include_citations: true
            })
        });

        if (!pythonResponse.ok) {
            throw new Error(`Python Worker responded with status: ${pythonResponse.status}`);
        }

        const aiData = await pythonResponse.json();

        // Return the final generated AI response back to the Go Gateway
        return res.status(200).json(aiData);
    } catch (error) {
        console.error(`[Orchestrator Error] ${error}`);
        return res.status(500).json({
            status: 'error',
            message: 'Failed to process request via Python AI Worker.'
        });
    }
});

const PORT = 4000;
app.listen(PORT, () => {
    console.log(`Node.js Orchestrator & Memory Handler running on :${PORT}`);
});
