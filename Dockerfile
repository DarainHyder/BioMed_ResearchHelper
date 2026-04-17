FROM python:3.10-slim

# Creates a non-root user that Hugging Face Spaces requires
RUN useradd -m -u 1000 user

WORKDIR /app

# Copy all project files into the container
COPY --chown=user . /app

# Temporarily alter the requirements.txt to strip exact versions
# This is identical to our Kaggle trick, allowing the Hugging Face
# container to pull the newest viable components effortlessly without crashing
RUN sed -i 's/==/>=/g' requirements.txt

# Install dependencies (ignoring the cache so the build stays light)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir faiss-cpu

# Switch to the non-root user
USER user

# Hugging Face Spaces assigns port 7860 dynamically
ENV API_PORT=7860
ENV API_HOST=0.0.0.0

EXPOSE 7860

# Launch the FastAPI app serving the models natively on 0.0.0.0
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
