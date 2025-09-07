FROM python:3.11-slim

WORKDIR /app

# Instala o Git (se precisar)
RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- ADICIONE ESTA LINHA AQUI ---
# Baixa o modelo Spacy durante a construção da imagem
RUN python -m spacy download pt_core_news_lg

COPY . .

EXPOSE 8000

# --- ESTA É A MUDANÇA MAIS IMPORTANTE ---
# Substitua o CMD antigo por este
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]