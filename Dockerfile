FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TICKETPILOT_PROJECT_ROOT=/app

WORKDIR /app

RUN addgroup --system ticketpilot \
    && adduser --system --ingroup ticketpilot --home /app ticketpilot

COPY pyproject.toml README.md ./
COPY src ./src
COPY docs ./docs

RUN python -m pip install --upgrade pip wheel \
    && python -m pip install --no-build-isolation .

RUN mkdir -p /app/artifacts/review \
    && chown -R ticketpilot:ticketpilot /app

USER ticketpilot

EXPOSE 8000

CMD ["uvicorn", "ticketpilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
