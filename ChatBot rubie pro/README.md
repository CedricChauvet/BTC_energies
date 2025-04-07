Pour le docker: (afin de ne pas rebuild tout le projet lors d'une modification de app.py)
# docker build -t rubie-pro-v12 .


solution a probleme de rebuild du docker a chaque fois qu'on modifie
app.py
# docker run -p 8080:8080-v "$(pwd)/app.py:/app/app.py" rubie-pro-v12



Suivre les  instructions suivantes:


# us-central1-docker.pkg.dev/btc-chatbot-pro/repo-chatbot

pour push le container Docker dans Artifact Registry
# docker tag vertex-ai-demo:latest europe-west9-docker.pkg.dev/btc-chatbot-pro/repo-chatbot/vertex-ai-demo:latest
# docker push europe-west9-docker.pkg.dev/btc-chatbot-pro/repo-chatbot/vertex-ai-demo:latest
# gcloud run deploy --image europe-west9-docker.pkg.dev/btc-chatbot-pro/repo-chatbot/vertex-ai-demo:latest --platform managed --memory 1Gi
# gcloud config set project btc-chatbot-pro

Création du Cluster

# configuration de base du cluster:
gcloud container clusters create rag-cluster \
    --zone europe-west1-b \
    --num-nodes 2 \
    --machine-type c2-standard-4 \
    --disk-size 100 \
    --enable-autoscaling \
    --min-nodes 1 \
    --max-nodes 3

puis executer:
# kubectl apply -f deployment.yaml (present sur le repoertoire)

Tests: Obtenir l'IP externe
# kubectl get service rag-cluster-hello-service

Une fois l'IP obtenue, testez avec curl. dois renvoyer des infos de index.html
# curl http://[IP_EXTERNE]

acces web
# http://[IP_EXTERNE]:80

Pour le log
# kubectl logs -l app=rag-cluster-hello -f