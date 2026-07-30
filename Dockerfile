FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code
COPY bot.py .

# Create log directory
RUN mkdir -p /tmp/bot_logs

# Expose port for log server
EXPOSE 10000

# Run the bot
CMD ["python", "bot.py"]
