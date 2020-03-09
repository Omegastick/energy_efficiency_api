# Train model
```bash
pip install -r requirements-train.txt
./train_model.py
```

# Run development server
```bash
export MODEL_PATH=$(pwd)/models/<Model you want to deploy>
sudo -E docker build . -t energy_efficiency_server
sudo -E docker run -it -v $MODEL_PATH:/model.pt -e MODEL_PATH=/model.pt -p 80:5000 --rm energy_efficiency_server ./run_server.py
```

# Run with uWSGI and Nginx
```bash
export MODEL_PATH=$(pwd)/models/<Model you want to deploy>
sudo -E docker-compose up --build
```

# Call with cURL
```bash
curl -H "Content-Type: application/json" --request POST --data '[0, 0, 0, 0, 0, 0, 0, 0]' localhost
```