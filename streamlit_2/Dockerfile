FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install poetry

# Install dependencies
RUN poetry install --no-root

# Copy the rest of the application code
COPY . .

# Default command to keep the container running
CMD ["tail", "-f", "/dev/null"]