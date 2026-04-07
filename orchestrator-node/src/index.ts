import express, { Request, Response } from 'express';

const app = express();
app.use(express.json());

// Multi-Turn Entity Preprocessor Mock
const resolveEntities = (input: string) => {
    // In reality: Check Working Memory for pronouns
    return input.replace(/he|she|it/g, "[resolved_entity]");
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

app.post('/api/analyze', (req: Request, res: Response) => {
    const { user_id, content } = req.body;
    console.log(`\n[Node Orchestrator] Processing input for ${user_id}`);

    // Step 1: Safety & Risk Escalation Guardrails
    const riskAssessment = checkRiskGuardrails(content);
    if (riskAssessment.riskFound) {
        console.warn(`[CRISIS ESCALATION] HIGH RISK DETECTED: Trigger="${riskAssessment.trigger}"`);
        // Antigravity action: Instantly route to crisis worker, halt standard processing.
        return res.status(200).json({ status: 'escalated', alert: 'Crisis intervention triggered.' });
    }

    // Step 2: Multi-Turn Entity Resolution
    const resolvedContent = resolveEntities(content);
    console.log(`[Node Orchestrator] Resolved Input: ${resolvedContent}`);

    // Step 3: Proceed to Semantic Knowledge Graph & AI prediction layers...
    // -> Enqueue via Redis/RabbitMQ to Python workers.

    res.status(200).json({ status: 'success', message: 'Input safely analyzed and queued.' });
});

const PORT = 4000;
app.listen(PORT, () => {
    console.log(`Node.js Orchestrator & Memory Handler running on :${PORT}`);
});
