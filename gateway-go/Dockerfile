# Stage 1: Build
FROM golang:alpine AS builder

WORKDIR /app

# Copy the go mod file, prepare download
COPY go.mod ./
RUN go mod download

# Copy the source code
COPY . .

# Build the executable
RUN CGO_ENABLED=0 GOOS=linux go build -o /omnigw cmd/server/main.go

# Stage 2: Minimal Distroless / Alpine
FROM alpine:latest
WORKDIR /app
COPY --from=builder /omnigw /app/omnigw

EXPOSE 8080

CMD ["/app/omnigw"]
