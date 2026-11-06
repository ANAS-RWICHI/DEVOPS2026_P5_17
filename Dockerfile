# One image for the three services (MLflow server, pipeline, API).

# ---- build stage: install pinned dependencies into a virtual environment
FROM python:3.12-slim AS builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# ---- runtime stage: only the virtual environment and the code
FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
COPY --from=builder /opt/venv /opt/venv

# The pipeline service bind-mounts the repository; let DVC read its .git
# even though the files belong to the host user.
RUN printf '[safe]\n\tdirectory = *\n' > /etc/gitconfig \
    && useradd --create-home app \
    && mkdir /mlflow && chown app:app /mlflow

WORKDIR /app
COPY --chown=app:app params.yaml dvc.yaml ./
COPY --chown=app:app src ./src

USER app
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
