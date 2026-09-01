#!/bin/bash
# KYT Engine deployment script for minikube

set -e

echo "=== KYT Engine Deployment ==="

# Start minikube
echo "Starting minikube..."
minikube start --memory=8192 --cpus=4 --driver=docker

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server

# Build Docker images inside minikube
echo "Building Docker images..."
eval $(minikube docker-env)
docker build -t kyt-engine:latest .

# Apply K8s manifests
echo "Deploying to Kubernetes..."
kubectl apply -f k8s/

# Wait for rollout
echo "Waiting for rollout..."
kubectl rollout status deployment/kyt-api -n kyt-engine --timeout=120s
kubectl rollout status deployment/kyt-dashboard -n kyt-engine --timeout=120s

# Get URLs
echo ""
echo "=== Deployment Complete ==="
echo "API: $(minikube service kyt-api -n kyt-engine --url 2>/dev/null)"
echo "Dashboard: $(minikube service kyt-dashboard -n kyt-engine --url 2>/dev/null)"
echo "MinIO Console: $(minikube service minio-console -n kyt-engine --url 2>/dev/null)"
