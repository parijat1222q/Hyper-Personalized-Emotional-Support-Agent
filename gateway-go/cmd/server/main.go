package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "os"
)

// IncomingInput represents diverse user inputs.
type IncomingInput struct {
    UserID    string `json:"user_id"`
    SessionID string `json:"session_id"`
    Content   string `json:"content"`
    Context   string `json:"context"` // e.g., "life context, casual"
}

// OmniInputHandler validates and routes to the Node.js orchestrator
func OmniInputHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var input IncomingInput
    if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
        http.Error(w, "Invalid input", http.StatusBadRequest)
        return
    }

    log.Printf("[Go Gateway] Received input from %s", input.UserID)

    // "Antigravity flow": Instantly forward to Node.js Orchestrator for guardrails & analysis.
    // In production, this would be gRPC. Using HTTP for simple boilerplate showcase.
    orchestratorURL := os.Getenv("ORCHESTRATOR_URL")
    if orchestratorURL == "" {
        orchestratorURL = "http://localhost:4000/api/analyze"
    }
    jsonData, _ := json.Marshal(input)

    resp, err := http.Post(orchestratorURL, "application/json", bytes.NewBuffer(jsonData))
    if err != nil {
        http.Error(w, "Orchestrator unreachable", http.StatusServiceUnavailable)
        return
    }
    defer resp.Body.Close()

    w.WriteHeader(http.StatusOK)
    w.Write([]byte("Request safely routed to orchestrator."))
}

func main() {
    http.HandleFunc("/api/intake", OmniInputHandler)
    fmt.Println("🚀 Go API Gateway running on :8080 (Omni-Input Handler active)")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
