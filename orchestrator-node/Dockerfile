FROM node:18-alpine

WORKDIR /app

# Install dependencies FIRST to leverage Docker cache
COPY package*.json ./
RUN npm install --include=dev

# Copy application logic
COPY . .

EXPOSE 4000

# Start script points to dev environment using npx for reliability
CMD ["npx", "nodemon", "src/index.ts"]
