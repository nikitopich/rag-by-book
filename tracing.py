import phoenix as px
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


def setup_tracing() -> str:
    """Запускает Phoenix UI и подключает трейсинг для OpenAI-клиента.
    Возвращает URL дашборда."""
    session = px.launch_app()

    endpoint = session.url + "v1/traces"
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint))
    )
    trace_api.set_tracer_provider(tracer_provider)

    OpenAIInstrumentor().instrument()

    return session.url
