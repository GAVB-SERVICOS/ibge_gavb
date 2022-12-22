#!/bin/bash
PROJECT_ID="gavb-poc-bu-mlops-f-store"
REGION="us-central1"
REPOSITORY="vertexai-teste"
IMAGE="hptune-vertxai:latest"

# Build image
docker build --tag=$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE .

# Create repository in the artifact registry
gcloud beta artifacts repositories create $REPOSITORY \
 --repository-format=docker \
 --location=$REGION


# Configure Docker
gcloud auth configure-docker $REGION-docker.pkg.dev


# Push
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE

#us-central1-docker.pkg.dev/gavb-poc-bu-mlops-f-store/vertexai-teste/hptune-vertxai