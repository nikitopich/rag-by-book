import socket

import phoenix as px
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

PHOENIX_HOST = "localhost"
PHOENIX_PORT = 6006
PHOENIX_URL = f"http://{PHOENIX_HOST}:{PHOENIX_PORT}/"


def _phoenix_already_running() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((PHOENIX_HOST, PHOENIX_PORT)) == 0


def setup_tracing() -> str:
    """Подключает Phoenix трейсинг. Если Phoenix уже запущен — переиспользует его.
    Возвращает URL дашборда."""
    if _phoenix_already_running():
        base_url = PHOENIX_URL
    else:
        session = px.launch_app()
        base_url = session.url

    endpoint = base_url + "v1/traces"
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint))
    )
    trace_api.set_tracer_provider(tracer_provider)

    OpenAIInstrumentor().instrument()

    return base_url
