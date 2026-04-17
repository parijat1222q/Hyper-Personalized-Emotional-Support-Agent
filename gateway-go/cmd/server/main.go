package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"time"
)

// ============================================================================
// TASK 3: HARDENED HTTP CLIENT WITH CONNECTION POOLING & TIMEOUTS
// ============================================================================

// IncomingInput represents diverse user inputs.
type IncomingInput struct {
	UserID    string `json:"user_id"`
	SessionID string `json:"session_id"`
	Content   string `json:"content"`
	Context   string `json:"context"` // e.g., "life context, casual"
}

// ErrorResponse represents standardized error responses
type ErrorResponse struct {
	Status  string `json:"status"`
	Error   string `json:"error"`
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// SuccessResponse represents successful routing responses
type SuccessResponse struct {
	Status   string `json:"status"`
	Message  string `json:"message"`
	UserID   string `json:"user_id"`
	Routed   bool   `json:"routed"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// GlobalHTTPClient is a production-grade HTTP client with connection pooling
var GlobalHTTPClient *http.Client

// initHTTPClient initializes a hardened HTTP client with:
// - Strict timeouts
// - Connection pooling
// - Custom dialer with TCP keep-alive
// - TLS handshake timeout
// - Expect-Continue timeout
func initHTTPClient() *http.Client {
	// Custom dialer with TCP keep-alive
	dialer := &net.Dialer{
		Timeout:   5 * time.Second,      // Connection establishment timeout
		KeepAlive: 30 * time.Second,     // TCP keep-alive interval
		DualStack: true,                 // Support IPv4 and IPv6
	}

	// Custom transport with connection pooling
	transport := &http.Transport{
		Dial:                dialer.Dial,
		DialContext:         dialer.DialContext,
		MaxIdleConns:        100,              // Maximum idle connections across all hosts
		MaxIdleConnsPerHost: 10,               // Maximum idle connections per host
		MaxConnsPerHost:     32,               // Maximum connections per host (prevents resource exhaustion)
		IdleConnTimeout:     90 * time.Second, // Close idle connections after 90s
		TLSHandshakeTimeout: 10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ResponseHeaderTimeout: 10 * time.Second, // Timeout for reading response headers
		DisableKeepAlives:   false,              // Keep connections alive for reuse
		DisableCompression:  false,              // Enable gzip compression
	}

	// Create custom client with transport and timeout
	client := &http.Client{
		Transport: transport,
		Timeout:   30 * time.Second, // Overall request timeout (increased for AI generation)
	}

	return client
}

// EnableCORS adds CORS headers to responses
func EnableCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Allow all origins (for development; restrict in production)
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
		w.Header().Set("Access-Control-Max-Age", "3600")

		// Handle preflight requests
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next(w, r)
	}
}

// OmniInputHandler validates and routes to the Node.js orchestrator with hardened HTTP client
func OmniInputHandler(w http.ResponseWriter, r *http.Request) {
	// Set response header as JSON
	w.Header().Set("Content-Type", "application/json")

	// Validate HTTP method
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(ErrorResponse{
			Status:  "error",
			Error:   "Invalid HTTP method",
			Code:    http.StatusMethodNotAllowed,
			Message: "Only POST requests are allowed",
		})
		return
	}

	// Parse incoming request
	var input IncomingInput
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{
			Status:  "error",
			Error:   "Invalid JSON payload",
			Code:    http.StatusBadRequest,
			Message: fmt.Sprintf("Failed to decode request body: %v", err),
		})
		log.Printf("[Go Gateway] ❌ Decode error: %v", err)
		return
	}

	// Validate required fields
	if input.UserID == "" || input.Content == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{
			Status:  "error",
			Error:   "Missing required fields",
			Code:    http.StatusBadRequest,
			Message: "user_id and content are required",
		})
		log.Printf("[Go Gateway] ❌ Missing required fields")
		return
	}

	log.Printf("[Go Gateway] ✓ Received input from %s (session: %s)", input.UserID, input.SessionID)

	// Get orchestrator URL from environment or use default
	orchestratorURL := os.Getenv("ORCHESTRATOR_URL")
	if orchestratorURL == "" {
		orchestratorURL = "http://orchestrator-node:4000/api/analyze"
	}

	// Marshal request body
	jsonData, err := json.Marshal(input)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{
			Status:  "error",
			Error:   "Internal server error",
			Code:    http.StatusInternalServerError,
			Message: "Failed to marshal request",
		})
		log.Printf("[Go Gateway] ❌ Marshal error: %v", err)
		return
	}

	
	// Create a context with a timeout for this specific request
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Create HTTP request with context
	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		orchestratorURL,
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{
			Status:  "error",
			Error:   "Failed to create request",
			Code:    http.StatusInternalServerError,
			Message: fmt.Sprintf("Request creation failed: %v", err),
		})
		log.Printf("[Go Gateway] ❌ Request creation error: %v", err)
		return
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Forwarded-For", getClientIP(r))
	req.Header.Set("User-Agent", "OmniMind-Gateway/1.0")

	// Execute request with hardened client
	resp, err := GlobalHTTPClient.Do(req)

	// ============================================================================
	// TASK 3: GRACEFUL ERROR HANDLING WITH APPROPRIATE STATUS CODES
	// ============================================================================

	if err != nil {
		// Determine error type and set appropriate status code
		statusCode := http.StatusServiceUnavailable // Default to 503
		errorMsg := "Orchestrator service unavailable"

		// Check for specific error types
		if ctx.Err() == context.DeadlineExceeded {
			statusCode = http.StatusGatewayTimeout // 504
			errorMsg = "Request to orchestrator timed out (30s limit exceeded)"
			log.Printf("[Go Gateway] ⏱️ TIMEOUT: Request to orchestrator exceeded deadline")
		} else if err, ok := err.(net.Error); ok && err.Timeout() {
			statusCode = http.StatusGatewayTimeout // 504
			errorMsg = "Connection to orchestrator timed out"
			log.Printf("[Go Gateway] ⏱️ TIMEOUT: Network timeout connecting to orchestrator")
		} else if err, ok := err.(net.Error); ok && err.Temporary() {
			statusCode = http.StatusServiceUnavailable // 503
			errorMsg = "Temporary failure communicating with orchestrator"
			log.Printf("[Go Gateway] 🔄 TEMPORARY ERROR: %v", err)
		} else {
			// Generic connection errors (connection refused, host unreachable, etc.)
			statusCode = http.StatusServiceUnavailable // 503
			errorMsg = "Cannot connect to orchestrator service"
			log.Printf("[Go Gateway] ❌ CONNECTION ERROR: %v", err)
		}

		w.WriteHeader(statusCode)
		json.NewEncoder(w).Encode(ErrorResponse{
			Status:  "error",
			Error:   "Orchestrator unreachable",
			Code:    statusCode,
			Message: errorMsg,
		})
		return
	}

	// Ensure response body is closed
	defer resp.Body.Close()

	// Check orchestrator response status
	if resp.StatusCode != http.StatusOK {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(ErrorResponse{
			Status:  "error",
			Error:   "Orchestrator returned error",
			Code:    resp.StatusCode,
			Message: fmt.Sprintf("Orchestrator responded with status %d", resp.StatusCode),
		})
		log.Printf("[Go Gateway] ❌ Orchestrator error response: %d", resp.StatusCode)
		return
	}

	// Success: Forward the orchestrator's response directly to the client
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	
	// Copy the orchestrator's response body directly to the client
	if _, err := io.Copy(w, resp.Body); err != nil {
		log.Printf("[Go Gateway] ⚠️ Error copying orchestrator response: %v", err)
	}

	log.Printf("[Go Gateway] ✅ Successfully routed request for user %s", input.UserID)
}

// getClientIP extracts client IP from request (handles X-Forwarded-For)
func getClientIP(r *http.Request) string {
	// Check for X-Forwarded-For header (proxy)
	if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
		return forwarded
	}
	// Check for X-Real-IP header
	if realIP := r.Header.Get("X-Real-IP"); realIP != "" {
		return realIP
	}
	// Fall back to RemoteAddr
	return r.RemoteAddr
}

// HealthCheckHandler provides health status of the gateway
func HealthCheckHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(ErrorResponse{
			Status:  "error",
			Error:   "Method not allowed",
			Code:    http.StatusMethodNotAllowed,
			Message: "Only GET requests allowed",
		})
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":     "healthy",
		"service":    "Go API Gateway",
		"version":    "1.0.0",
		"uptime":     "running",
		"timestamp":  time.Now().Unix(),
		"http_client": map[string]interface{}{
			"max_idle_conns":         100,
			"max_idle_conns_per_host": 10,
			"max_conns_per_host":     32,
			"timeout":                "30s",
		},
	})

	log.Printf("[Go Gateway] ✓ Health check passed")
}

func main() {
	// Initialize the hardened HTTP client once at startup
	GlobalHTTPClient = initHTTPClient()
	log.Println("[Go Gateway] 🔧 Hardened HTTP client initialized")
	log.Printf("[Go Gateway] ✓ Connection Pool: MaxIdleConns=100, MaxIdleConnsPerHost=10, MaxConnsPerHost=32")
	log.Printf("[Go Gateway] ✓ Timeouts: Overall=30s, Dial=5s, TLSHandshake=10s")

	// Register handlers with CORS support
	http.HandleFunc("/api/intake", EnableCORS(OmniInputHandler))
	http.HandleFunc("/health", EnableCORS(HealthCheckHandler))

	// Start the gateway
	fmt.Println("🚀 Go API Gateway running on :8080")
	fmt.Println("   📌 /api/intake - Primary input handler (POST)")
	fmt.Println("   📌 /health - Gateway health check (GET)")
	fmt.Println("")

	// Use ListenAndServe with a configured server for production readiness
	server := &http.Server{
		Addr:         ":8080",
		Handler:      nil, // Use DefaultServeMux
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  90 * time.Second,
	}

	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("❌ Server failed to start: %v", err)
	}
}
