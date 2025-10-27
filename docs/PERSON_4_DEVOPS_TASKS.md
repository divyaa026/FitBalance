# Person 4: DevOps & Deployment Engineer - Task Guide

**Project:** FitBalance  
**Duration:** 1.5-2 weeks (40-50 hours)  
**Contribution:** 15-20% of total project  
**Role:** EXCLUSIVE DEVOPS/DEPLOYMENT WORK - CI/CD, Docker, Cloud Infrastructure

---

## 📋 Overview

You are responsible for **ALL deployment and infrastructure work**:
1. **Containerization** (Docker for backend, frontend, databases)
2. **CI/CD Pipeline** (GitHub Actions for automated testing & deployment)
3. **Cloud Deployment** (AWS/Azure/GCP setup)
4. **Database Setup** (PostgreSQL with migrations)
5. **Monitoring & Logging** (Application health checks)
6. **Documentation** (Deployment guides, runbooks)

**No frontend work, no backend logic, no ML training** - focus 100% on infrastructure and deployment.

---

## 🎯 Your Deliverables

- ✅ Docker containers for all services
- ✅ Docker Compose for local development
- ✅ CI/CD pipeline with automated tests
- ✅ Cloud deployment (production-ready)
- ✅ PostgreSQL database setup with migrations
- ✅ Environment variable management
- ✅ SSL/HTTPS configuration
- ✅ Monitoring and logging setup
- ✅ Deployment documentation
- ✅ Disaster recovery plan

---

## 📂 File Structure You'll Create

```
FitBalance/
├── .github/
│   └── workflows/
│       ├── backend-ci.yml          # ← YOU: Create
│       ├── frontend-ci.yml         # ← YOU: Create
│       └── deploy-prod.yml         # ← YOU: Create
├── docker/
│   ├── backend/
│   │   └── Dockerfile              # ← YOU: Create
│   ├── frontend/
│   │   └── Dockerfile              # ← YOU: Create
│   └── nginx/
│       ├── Dockerfile              # ← YOU: Create
│       └── nginx.conf              # ← YOU: Create
├── docker-compose.yml              # ← YOU: Create
├── docker-compose.prod.yml         # ← YOU: Create
├── .env.example                    # ← YOU: Create
├── scripts/
│   ├── setup-dev.sh                # ← YOU: Create
│   ├── deploy-prod.sh              # ← YOU: Create
│   ├── backup-db.sh                # ← YOU: Create
│   └── restore-db.sh               # ← YOU: Create
├── kubernetes/                     # ← YOU: Optional (K8s)
│   ├── backend-deployment.yaml
│   ├── frontend-deployment.yaml
│   └── postgres-deployment.yaml
└── docs/
    ├── DEPLOYMENT_GUIDE.md         # ← YOU: Create
    ├── MONITORING_GUIDE.md         # ← YOU: Create
    └── DISASTER_RECOVERY.md        # ← YOU: Create
```

---

## 🚀 TASK 1: Docker Containerization
**Time:** 12 hours

### Step 1.1: Create Backend Dockerfile (3 hours)

**Create:** `docker/backend/Dockerfile`

```dockerfile
# Multi-stage build for Python backend
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY backend/ ./backend/
COPY ml_models/ ./ml_models/
COPY datasets/ ./datasets/
COPY start_server.py .

# Make sure scripts are executable
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Test locally:**
```powershell
cd c:\Users\divya\Desktop\projects\FitBalance

# Build image
docker build -f docker/backend/Dockerfile -t fitbalance-backend:latest .

# Run container
docker run -p 8000:8000 fitbalance-backend:latest

# Test in another terminal
curl http://localhost:8000/health
```

---

### Step 1.2: Create Frontend Dockerfile (3 hours)

**Create:** `docker/frontend/Dockerfile`

```dockerfile
# Multi-stage build for React frontend
FROM node:20-alpine as builder

WORKDIR /app

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy source code
COPY frontend/ .

# Build the application
RUN npm run build

# Production stage with Nginx
FROM nginx:alpine

# Copy custom nginx config
COPY docker/nginx/nginx.conf /etc/nginx/nginx.conf

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
```

**Create:** `docker/nginx/nginx.conf`

```nginx
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript 
               application/x-javascript application/xml+rss 
               application/json application/javascript;

    server {
        listen 80;
        server_name localhost;
        root /usr/share/nginx/html;
        index index.html;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;

        # Frontend routes
        location / {
            try_files $uri $uri/ /index.html;
        }

        # API proxy to backend
        location /api/ {
            proxy_pass http://backend:8000/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

**Test locally:**
```powershell
# Build image
docker build -f docker/frontend/Dockerfile -t fitbalance-frontend:latest .

# Run container
docker run -p 8081:80 fitbalance-frontend:latest

# Open browser: http://localhost:8081
```

---

### Step 1.3: Create Docker Compose (6 hours)

**Create:** `docker-compose.yml` (Development)

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: fitbalance-db
    environment:
      POSTGRES_DB: fitbalance
      POSTGRES_USER: fitbalance_user
      POSTGRES_PASSWORD: fitbalance_dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fitbalance_user -d fitbalance"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - fitbalance-network

  # Backend API
  backend:
    build:
      context: .
      dockerfile: docker/backend/Dockerfile
    container_name: fitbalance-backend
    environment:
      - DATABASE_URL=postgresql://fitbalance_user:fitbalance_dev_password@postgres:5432/fitbalance
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
      - CORS_ORIGINS=http://localhost:8081,http://localhost:5173
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app/backend
      - ./ml_models:/app/ml_models
      - ./datasets:/app/datasets
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - fitbalance-network
    restart: unless-stopped

  # Frontend
  frontend:
    build:
      context: .
      dockerfile: docker/frontend/Dockerfile
    container_name: fitbalance-frontend
    environment:
      - VITE_API_URL=http://localhost:8000
    ports:
      - "8081:80"
    depends_on:
      - backend
    networks:
      - fitbalance-network
    restart: unless-stopped

  # Redis (for caching)
  redis:
    image: redis:7-alpine
    container_name: fitbalance-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - fitbalance-network
    restart: unless-stopped

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local

networks:
  fitbalance-network:
    driver: bridge
```

**Create:** `docker-compose.prod.yml` (Production)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: fitbalance-db-prod
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - fitbalance-network
    restart: always

  backend:
    image: fitbalance-backend:${VERSION:-latest}
    container_name: fitbalance-backend-prod
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - SECRET_KEY=${SECRET_KEY}
      - CORS_ORIGINS=${CORS_ORIGINS}
      - REDIS_URL=redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - fitbalance-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  frontend:
    image: fitbalance-frontend:${VERSION:-latest}
    container_name: fitbalance-frontend-prod
    environment:
      - VITE_API_URL=${API_URL}
    depends_on:
      - backend
    networks:
      - fitbalance-network
    restart: always

  redis:
    image: redis:7-alpine
    container_name: fitbalance-redis-prod
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - fitbalance-network
    restart: always

  # Nginx reverse proxy (production)
  nginx:
    image: nginx:alpine
    container_name: fitbalance-nginx-prod
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - frontend
      - backend
    networks:
      - fitbalance-network
    restart: always

volumes:
  postgres_data:
  redis_data:

networks:
  fitbalance-network:
    driver: bridge
```

**Create:** `.env.example`

```env
# Database
POSTGRES_DB=fitbalance
POSTGRES_USER=fitbalance_user
POSTGRES_PASSWORD=change_me_in_production

# Backend
SECRET_KEY=generate_secure_key_here
ENVIRONMENT=production
LOG_LEVEL=INFO
CORS_ORIGINS=https://yourdomain.com

# Redis
REDIS_PASSWORD=change_me_in_production

# API
API_URL=https://api.yourdomain.com

# Version
VERSION=1.0.0
```

**Test Docker Compose:**
```powershell
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

---

## 🔄 TASK 2: CI/CD Pipeline with GitHub Actions
**Time:** 10 hours

### Step 2.1: Backend CI Pipeline (3 hours)

**Create:** `.github/workflows/backend-ci.yml`

```yaml
name: Backend CI

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'backend/**'
      - 'ml_models/**'
      - 'requirements.txt'
      - '.github/workflows/backend-ci.yml'
  pull_request:
    branches: [ main, develop ]
    paths:
      - 'backend/**'
      - 'ml_models/**'
      - 'requirements.txt'

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: fitbalance_test
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_password
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python 3.12
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio

    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 backend/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 backend/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    - name: Type check with mypy
      run: |
        pip install mypy
        mypy backend/ --ignore-missing-imports || true

    - name: Run tests with pytest
      env:
        DATABASE_URL: postgresql://test_user:test_password@localhost:5432/fitbalance_test
      run: |
        pytest backend/ --cov=backend --cov-report=xml --cov-report=html -v

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: backend
        name: backend-coverage

    - name: Test backend startup
      run: |
        timeout 10s python start_server.py || code=$?; if [[ $code -ne 124 && $code -ne 0 ]]; then exit $code; fi

  build:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./docker/backend/Dockerfile
        push: true
        tags: |
          ${{ secrets.DOCKER_USERNAME }}/fitbalance-backend:latest
          ${{ secrets.DOCKER_USERNAME }}/fitbalance-backend:${{ github.sha }}
        cache-from: type=registry,ref=${{ secrets.DOCKER_USERNAME }}/fitbalance-backend:buildcache
        cache-to: type=registry,ref=${{ secrets.DOCKER_USERNAME }}/fitbalance-backend:buildcache,mode=max
```

---

### Step 2.2: Frontend CI Pipeline (3 hours)

**Create:** `.github/workflows/frontend-ci.yml`

```yaml
name: Frontend CI

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'frontend/**'
      - '.github/workflows/frontend-ci.yml'
  pull_request:
    branches: [ main, develop ]
    paths:
      - 'frontend/**'

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '20'
        cache: 'npm'
        cache-dependency-path: frontend/package-lock.json

    - name: Install dependencies
      working-directory: ./frontend
      run: npm ci

    - name: Lint code
      working-directory: ./frontend
      run: npm run lint || true

    - name: Type check
      working-directory: ./frontend
      run: npx tsc --noEmit

    - name: Run tests
      working-directory: ./frontend
      run: npm test -- --run --coverage

    - name: Build application
      working-directory: ./frontend
      env:
        VITE_API_URL: https://api.fitbalance.com
      run: npm run build

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        directory: ./frontend/coverage
        flags: frontend
        name: frontend-coverage

    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: frontend-dist
        path: frontend/dist

  build:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./docker/frontend/Dockerfile
        push: true
        tags: |
          ${{ secrets.DOCKER_USERNAME }}/fitbalance-frontend:latest
          ${{ secrets.DOCKER_USERNAME }}/fitbalance-frontend:${{ github.sha }}
        cache-from: type=registry,ref=${{ secrets.DOCKER_USERNAME }}/fitbalance-frontend:buildcache
        cache-to: type=registry,ref=${{ secrets.DOCKER_USERNAME }}/fitbalance-frontend:buildcache,mode=max
```

---

### Step 2.3: Production Deployment Pipeline (4 hours)

**Create:** `.github/workflows/deploy-prod.yml`

```yaml
name: Deploy to Production

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to deploy'
        required: true
        default: 'latest'

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1

    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1

    - name: Deploy to EC2
      env:
        PRIVATE_KEY: ${{ secrets.EC2_SSH_PRIVATE_KEY }}
        HOST: ${{ secrets.EC2_HOST }}
        USER: ${{ secrets.EC2_USER }}
        VERSION: ${{ github.event.inputs.version }}
      run: |
        echo "$PRIVATE_KEY" > private_key.pem
        chmod 600 private_key.pem
        
        ssh -o StrictHostKeyChecking=no -i private_key.pem ${USER}@${HOST} << 'EOF'
          cd /home/ubuntu/fitbalance
          
          # Pull latest code
          git pull origin main
          
          # Set environment variables
          export VERSION=${{ github.event.inputs.version }}
          
          # Pull latest Docker images
          docker-compose -f docker-compose.prod.yml pull
          
          # Stop old containers
          docker-compose -f docker-compose.prod.yml down
          
          # Start new containers
          docker-compose -f docker-compose.prod.yml up -d
          
          # Run database migrations
          docker-compose -f docker-compose.prod.yml exec -T backend alembic upgrade head
          
          # Health check
          sleep 10
          curl -f http://localhost:8000/health || exit 1
          
          # Clean up old images
          docker image prune -f
        EOF
        
        rm private_key.pem

    - name: Verify deployment
      run: |
        sleep 30
        curl -f https://api.fitbalance.com/health || exit 1
        curl -f https://fitbalance.com || exit 1

    - name: Notify Slack
      if: always()
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        text: 'Production deployment ${{ job.status }}'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## ☁️ TASK 3: Cloud Deployment (AWS)
**Time:** 10 hours

### Step 3.1: AWS Infrastructure Setup (5 hours)

**Create:** `scripts/aws-setup.sh`

```bash
#!/bin/bash
# AWS Infrastructure Setup Script

set -e

echo "=== FitBalance AWS Infrastructure Setup ==="

# Configuration
REGION="us-east-1"
APP_NAME="fitbalance"
KEY_NAME="fitbalance-key"

# Create VPC
echo "Creating VPC..."
VPC_ID=$(aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --region $REGION \
  --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=$APP_NAME-vpc}]" \
  --query 'Vpc.VpcId' \
  --output text)

echo "VPC ID: $VPC_ID"

# Create Internet Gateway
echo "Creating Internet Gateway..."
IGW_ID=$(aws ec2 create-internet-gateway \
  --region $REGION \
  --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=$APP_NAME-igw}]" \
  --query 'InternetGateway.InternetGatewayId' \
  --output text)

aws ec2 attach-internet-gateway \
  --vpc-id $VPC_ID \
  --internet-gateway-id $IGW_ID \
  --region $REGION

# Create Subnet
echo "Creating Subnet..."
SUBNET_ID=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.1.0/24 \
  --availability-zone ${REGION}a \
  --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=$APP_NAME-subnet}]" \
  --query 'Subnet.SubnetId' \
  --output text)

# Create Route Table
echo "Creating Route Table..."
ROUTE_TABLE_ID=$(aws ec2 create-route-table \
  --vpc-id $VPC_ID \
  --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=$APP_NAME-rt}]" \
  --query 'RouteTable.RouteTableId' \
  --output text)

aws ec2 create-route \
  --route-table-id $ROUTE_TABLE_ID \
  --destination-cidr-block 0.0.0.0/0 \
  --gateway-id $IGW_ID \
  --region $REGION

aws ec2 associate-route-table \
  --subnet-id $SUBNET_ID \
  --route-table-id $ROUTE_TABLE_ID \
  --region $REGION

# Create Security Group
echo "Creating Security Group..."
SG_ID=$(aws ec2 create-security-group \
  --group-name $APP_NAME-sg \
  --description "Security group for FitBalance application" \
  --vpc-id $VPC_ID \
  --region $REGION \
  --query 'GroupId' \
  --output text)

# Add security group rules
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0 \
  --region $REGION

aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0 \
  --region $REGION

aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0 \
  --region $REGION

aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 8000 \
  --cidr 0.0.0.0/0 \
  --region $REGION

# Create Key Pair
echo "Creating Key Pair..."
aws ec2 create-key-pair \
  --key-name $KEY_NAME \
  --region $REGION \
  --query 'KeyMaterial' \
  --output text > ${KEY_NAME}.pem

chmod 400 ${KEY_NAME}.pem

echo "=== Infrastructure created successfully! ==="
echo "VPC ID: $VPC_ID"
echo "Subnet ID: $SUBNET_ID"
echo "Security Group ID: $SG_ID"
echo "Key saved to: ${KEY_NAME}.pem"
```

**Create:** `scripts/deploy-ec2.sh`

```bash
#!/bin/bash
# Deploy FitBalance to EC2

set -e

echo "=== Deploying FitBalance to EC2 ==="

# Configuration
REGION="us-east-1"
INSTANCE_TYPE="t3.large"
AMI_ID="ami-0c55b159cbfafe1f0"  # Ubuntu 22.04 LTS
KEY_NAME="fitbalance-key"
SG_ID="sg-xxxxx"  # Replace with your security group ID
SUBNET_ID="subnet-xxxxx"  # Replace with your subnet ID

# Launch EC2 instance
echo "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --count 1 \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SG_ID \
  --subnet-id $SUBNET_ID \
  --region $REGION \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=fitbalance-prod}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance ID: $INSTANCE_ID"

# Wait for instance to be running
echo "Waiting for instance to start..."
aws ec2 wait instance-running \
  --instance-ids $INSTANCE_ID \
  --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --region $REGION \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Public IP: $PUBLIC_IP"

# Wait for SSH to be ready
echo "Waiting for SSH..."
sleep 60

# SSH and setup
echo "Setting up server..."
ssh -o StrictHostKeyChecking=no -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP} << 'EOF'
  # Update system
  sudo apt-get update
  sudo apt-get upgrade -y

  # Install Docker
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker ubuntu

  # Install Docker Compose
  sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose

  # Install Git
  sudo apt-get install -y git

  # Clone repository
  git clone https://github.com/yourusername/FitBalance.git
  cd FitBalance

  # Create .env file
  echo "POSTGRES_DB=fitbalance" > .env
  echo "POSTGRES_USER=fitbalance_user" >> .env
  echo "POSTGRES_PASSWORD=$(openssl rand -base64 32)" >> .env
  echo "SECRET_KEY=$(openssl rand -base64 32)" >> .env
  echo "REDIS_PASSWORD=$(openssl rand -base64 32)" >> .env

  # Start services
  docker-compose -f docker-compose.prod.yml up -d

  echo "Deployment complete!"
EOF

echo "=== Deployment successful! ==="
echo "Access your application at: http://${PUBLIC_IP}"
echo "Save this information:"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
```

---

### Step 3.2: Database Setup & Migrations (5 hours)

**Create:** `scripts/init-db.sql`

```sql
-- Initial database setup
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create nutrition_logs table
CREATE TABLE IF NOT EXISTS nutrition_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    meal_image_url TEXT,
    detected_foods JSONB,
    total_protein DECIMAL(10, 2),
    total_calories DECIMAL(10, 2),
    recommendations TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create biomechanics_logs table
CREATE TABLE IF NOT EXISTS biomechanics_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    exercise_type VARCHAR(100),
    video_url TEXT,
    form_score DECIMAL(5, 2),
    joint_angles JSONB,
    risk_factors TEXT[],
    recommendations TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create burnout_assessments table
CREATE TABLE IF NOT EXISTS burnout_assessments (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    workout_frequency INTEGER,
    sleep_hours DECIMAL(3, 1),
    stress_level INTEGER,
    recovery_time INTEGER,
    risk_level VARCHAR(20),
    risk_score DECIMAL(5, 2),
    survival_probability DECIMAL(5, 4),
    recommendations TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_nutrition_user_created ON nutrition_logs(user_id, created_at DESC);
CREATE INDEX idx_biomechanics_user_created ON biomechanics_logs(user_id, created_at DESC);
CREATE INDEX idx_burnout_user_created ON burnout_assessments(user_id, created_at DESC);

-- Insert sample user for testing
INSERT INTO users (email, username, password_hash) 
VALUES ('demo@fitbalance.com', 'demo_user', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5U7vLEWvQV0.y')
ON CONFLICT (email) DO NOTHING;
```

**Create:** `scripts/backup-db.sh`

```bash
#!/bin/bash
# Database backup script

set -e

BACKUP_DIR="/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/fitbalance_backup_$TIMESTAMP.sql"

echo "Starting database backup..."

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Backup database
docker-compose exec -T postgres pg_dump -U fitbalance_user fitbalance > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

echo "Backup completed: ${BACKUP_FILE}.gz"

# Keep only last 7 days of backups
find $BACKUP_DIR -name "fitbalance_backup_*.sql.gz" -mtime +7 -delete

echo "Old backups cleaned up"
```

**Create:** `scripts/restore-db.sh`

```bash
#!/bin/bash
# Database restore script

set -e

if [ -z "$1" ]; then
    echo "Usage: ./restore-db.sh <backup_file.sql.gz>"
    exit 1
fi

BACKUP_FILE=$1

echo "Restoring database from $BACKUP_FILE..."

# Decompress if needed
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | docker-compose exec -T postgres psql -U fitbalance_user fitbalance
else
    cat $BACKUP_FILE | docker-compose exec -T postgres psql -U fitbalance_user fitbalance
fi

echo "Database restored successfully"
```

---

## 📊 TASK 4: Monitoring & Logging
**Time:** 6 hours

### Step 4.1: Add Health Checks (2 hours)

**Update:** `backend/main.py`

Add health check endpoint:

```python
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with dependency status"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check database
    try:
        # Add your database check here
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        # Add your Redis check here
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status
```

---

### Step 4.2: Configure Logging (2 hours)

**Create:** `backend/logging_config.py`

```python
import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    """Configure application logging"""
    
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # File handler with rotation
            RotatingFileHandler(
                "logs/fitbalance.log",
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    
    return logging.getLogger(__name__)

logger = setup_logging()
```

---

### Step 4.3: Monitoring Dashboard Setup (2 hours)

**Create:** `docker-compose.monitoring.yml`

```yaml
version: '3.8'

services:
  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    container_name: fitbalance-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - fitbalance-network

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: fitbalance-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana-dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    networks:
      - fitbalance-network

volumes:
  prometheus_data:
  grafana_data:

networks:
  fitbalance-network:
    external: true
```

**Create:** `monitoring/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fitbalance-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
```

---

## 📚 TASK 5: Documentation
**Time:** 4 hours

### Step 5.1: Deployment Guide

**Create:** `docs/DEPLOYMENT_GUIDE.md`

```markdown
# FitBalance Deployment Guide

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- AWS CLI configured
- Domain name configured (for production)
- SSL certificates (for production)

## Local Development

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/FitBalance.git
cd FitBalance

# Copy environment file
cp .env.example .env

# Edit .env with your settings
nano .env

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Access Services

- Frontend: http://localhost:8081
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- PostgreSQL: localhost:5432
- Redis: localhost:6379

## Production Deployment

### Option 1: AWS EC2

```bash
# 1. Setup AWS infrastructure
./scripts/aws-setup.sh

# 2. Deploy to EC2
./scripts/deploy-ec2.sh

# 3. Configure DNS
# Point your domain to the EC2 public IP

# 4. Setup SSL with Let's Encrypt
ssh -i fitbalance-key.pem ubuntu@YOUR_IP
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

### Option 2: Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml fitbalance

# Check services
docker stack services fitbalance
```

### Option 3: Kubernetes

```bash
# Apply configurations
kubectl apply -f kubernetes/

# Check deployment
kubectl get pods -n fitbalance
kubectl get services -n fitbalance
```

## Database Management

### Backup

```bash
# Manual backup
./scripts/backup-db.sh

# Automated backups (add to crontab)
0 2 * * * /path/to/FitBalance/scripts/backup-db.sh
```

### Restore

```bash
./scripts/restore-db.sh /backups/fitbalance_backup_YYYYMMDD_HHMMSS.sql.gz
```

### Migrations

```bash
# Run migrations
docker-compose exec backend alembic upgrade head

# Rollback
docker-compose exec backend alembic downgrade -1
```

## Monitoring

### Access Monitoring Tools

- Grafana: http://your-server:3000 (admin/admin)
- Prometheus: http://your-server:9090

### View Logs

```bash
# All logs
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend
```

## Troubleshooting

### Backend won't start

```bash
# Check logs
docker-compose logs backend

# Restart service
docker-compose restart backend

# Rebuild image
docker-compose build backend
docker-compose up -d backend
```

### Database connection issues

```bash
# Check database is running
docker-compose ps postgres

# Test connection
docker-compose exec postgres psql -U fitbalance_user -d fitbalance

# Reset database
docker-compose down -v
docker-compose up -d
```

### Frontend not loading

```bash
# Check nginx logs
docker-compose logs frontend

# Verify API URL
docker-compose exec frontend cat /usr/share/nginx/html/assets/*.js | grep API_URL
```

## Security Checklist

- [ ] Change all default passwords
- [ ] Setup SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Enable CORS only for your domain
- [ ] Setup automated backups
- [ ] Enable monitoring and alerts
- [ ] Review security group rules
- [ ] Setup WAF (Web Application Firewall)

## Scaling

### Horizontal Scaling

```bash
# Scale backend
docker-compose up -d --scale backend=3

# Scale with load balancer
# Add nginx upstream configuration
```

### Vertical Scaling

```yaml
# Update docker-compose.prod.yml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
```

## Maintenance

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild images
docker-compose build

# Rolling update
docker-compose up -d
```

### Clean Up

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove unused containers
docker container prune
```
```

---

### Step 5.2: Monitoring Guide

**Create:** `docs/MONITORING_GUIDE.md`

```markdown
# FitBalance Monitoring Guide

## Overview

FitBalance uses the following monitoring stack:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **Docker logs**: Application logs

## Key Metrics to Monitor

### Application Health

- **HTTP Request Rate**: Requests per second
- **Response Time**: P50, P95, P99 latency
- **Error Rate**: 4xx and 5xx responses
- **Active Users**: Concurrent connections

### System Resources

- **CPU Usage**: Per container
- **Memory Usage**: RSS and cache
- **Disk I/O**: Read/write operations
- **Network I/O**: Bandwidth usage

### Database

- **Connection Pool**: Active connections
- **Query Performance**: Slow queries
- **Database Size**: Growth over time
- **Replication Lag**: If using replicas

## Alerts

### Critical Alerts

1. **Service Down**
   - Trigger: Health check fails
   - Action: Auto-restart, notify team

2. **High Error Rate**
   - Trigger: >5% 5xx errors
   - Action: Check logs, investigate

3. **Database Down**
   - Trigger: Connection failures
   - Action: Check DB status, restore from backup

### Warning Alerts

1. **High CPU Usage**
   - Trigger: >80% for 5 minutes
   - Action: Consider scaling

2. **Low Disk Space**
   - Trigger: <10% free
   - Action: Clean up old logs/backups

3. **Slow Responses**
   - Trigger: P95 >2 seconds
   - Action: Optimize queries

## Grafana Dashboards

### Application Dashboard

Metrics:
- Request rate
- Response time
- Error rate
- Active users

### Infrastructure Dashboard

Metrics:
- CPU per container
- Memory per container
- Disk usage
- Network throughput

### Database Dashboard

Metrics:
- Connections
- Query performance
- Cache hit rate
- Transaction rate

## Log Analysis

### View Logs

```bash
# All backend logs
docker-compose logs -f backend

# Search for errors
docker-compose logs backend | grep ERROR

# Last hour
docker-compose logs --since 1h backend
```

### Common Log Patterns

```bash
# Database errors
grep "database" logs/fitbalance.log

# API errors
grep "ERROR" logs/fitbalance.log | grep "/api/"

# Slow queries
grep "slow" logs/fitbalance.log
```

## Performance Optimization

### Backend

1. Enable response caching
2. Use connection pooling
3. Optimize database queries
4. Use async operations

### Frontend

1. Enable Gzip compression
2. Use CDN for static assets
3. Implement lazy loading
4. Optimize images

### Database

1. Add indexes
2. Optimize queries
3. Regular VACUUM
4. Connection pooling
```

---

## ✅ Final Checklist

### Docker & Containerization
- [ ] Backend Dockerfile created and tested
- [ ] Frontend Dockerfile created and tested
- [ ] Nginx configuration working
- [ ] Docker Compose (dev) works
- [ ] Docker Compose (prod) works
- [ ] All containers have health checks

### CI/CD Pipeline
- [ ] Backend CI pipeline configured
- [ ] Frontend CI pipeline configured
- [ ] Production deployment pipeline working
- [ ] Docker images building successfully
- [ ] Tests running in CI
- [ ] Automated deployments working

### Cloud Deployment
- [ ] AWS infrastructure set up
- [ ] EC2 instance deployed
- [ ] Security groups configured
- [ ] Domain DNS configured
- [ ] SSL certificates installed
- [ ] Load balancer configured (if applicable)

### Database
- [ ] PostgreSQL running
- [ ] Initial schema created
- [ ] Migrations working
- [ ] Backup script tested
- [ ] Restore script tested
- [ ] Automated backups scheduled

### Monitoring
- [ ] Health check endpoints working
- [ ] Logging configured
- [ ] Prometheus collecting metrics
- [ ] Grafana dashboards created
- [ ] Alerts configured
- [ ] Log rotation enabled

### Documentation
- [ ] Deployment guide complete
- [ ] Monitoring guide complete
- [ ] Disaster recovery plan documented
- [ ] Runbooks created
- [ ] Environment variables documented

---

## 📊 Success Metrics

Your work is complete when:
1. ✅ Application deploys with one command
2. ✅ CI/CD pipeline runs automatically
3. ✅ Production environment is stable
4. ✅ Monitoring shows all services healthy
5. ✅ Backups are automated
6. ✅ Documentation is comprehensive

---

## 🆘 Getting Help

If stuck:
1. **Check logs**: `docker-compose logs -f`
2. **Test locally**: Always test in dev first
3. **Review docs**: AWS, Docker, GitHub Actions docs
4. **Ask team**: Coordinate with Person 2 (Backend) and Person 3 (Frontend)

---

## 📅 Timeline

**Week 1:**
- Days 1-2: Docker containerization
- Days 3-4: CI/CD pipeline setup
- Day 5: Testing and debugging

**Week 2:**
- Days 1-2: Cloud deployment (AWS)
- Days 3: Database & monitoring
- Days 4-5: Documentation & polish

---

## 🎉 Completion

Once done:
1. Deploy to production
2. Run health checks
3. Monitor for 24 hours
4. Share access with team
5. Conduct deployment demo

**Your contribution: 15-20% of total project** 🏆

This is critical infrastructure work - the entire team depends on you! 🚀
