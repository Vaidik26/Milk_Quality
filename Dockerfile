# Use official Python image
FROM python:3.10-slim


# Set working directory in the container
WORKDIR /app

# Copy project files into the container
COPY . .

# Set PYTHONPATH so imports from src/milk_quality work
ENV PYTHONPATH=/app/src

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (default Flask port)
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]

