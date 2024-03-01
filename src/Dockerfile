# Use the official lightweight Python image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python requirements file into the container
COPY app_requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r app_requirements.txt

# Copy the application code into the container
COPY app_v1.py .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "app_v1.py"]