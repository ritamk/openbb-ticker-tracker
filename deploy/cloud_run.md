# Cloud Run Deployment Guide

## 0. Quick deployment script

Prefer the helper script for repeatable builds:

```bash
chmod +x deploy/cloud_run_deploy.sh
./deploy/cloud_run_deploy.sh \
  --project openbb-py \
  --region us-central1 \
  --env-file deploy/cloud_run.env
```

Flags map directly to the `gcloud` commands below. Use `--help` for all options (tests, skipping builds, toggling unauth access, extra deploy args, etc.).

## 1. Configure Google Cloud project
```bash
gcloud auth login
gcloud config set project openbb-py
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
```

## 2. Build and push container image
```bash
REGION=us-central1
SERVICE=trading-llm-api
IMAGE=gcr.io/$(gcloud config get-value project)/${SERVICE}

gcloud builds submit --tag ${IMAGE}
```

## 3. Deploy to Cloud Run (fully managed)
```bash
gcloud run deploy ${SERVICE} \
  --image ${IMAGE} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --port 8080
```

Environment variables can be injected at deploy-time:
```bash
gcloud run deploy ${SERVICE} \
  --image ${IMAGE} \
  --platform managed \
  --region ${REGION} \
  --set-env-vars OPENAI_API_KEY=<VALUE>,TRADING_API_TRADES_BACKGROUND=true
```

## 4. (Optional) Firebase Hosting rewrite
1. Initialize Firebase Hosting inside the repo root:
   ```bash
   firebase init hosting
   ```
   - Choose an existing project or create a new one.
   - Select "Configure as a single-page app" = **No**.

2. Update `firebase.json` rewrites:
   ```json
   {
     "hosting": {
       "rewrites": [
         {
           "source": "/api/**",
           "run": {
             "serviceId": "trading-llm-api",
             "region": "us-central1"
           }
         }
       ]
     }
   }
   ```

3. Deploy hosting:
   ```bash
   firebase deploy --only hosting
   ```

Requests to `/api/*` on the Firebase-hosted domain will now proxy to the Cloud Run service.

## 5. Post-deploy checklist
- Store secrets (e.g., `OPENAI_API_KEY`) in Secret Manager and mount them via `--set-secrets`.
- Configure Cloud Logging alerts for error spikes.
- Restrict unauthenticated access if needed (Cloud Run IAM or Cloud Armor).

