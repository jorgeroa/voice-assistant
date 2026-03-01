"""Voice Conversation App - Entry Point"""

import logging
import sys

import click
import uvicorn

from config.settings import AppSettings
from server.app import app, configure


@click.command()
@click.option("--host", default=None, help="Server host")
@click.option("--port", default=None, type=int, help="Server port")
@click.option("--language", "-l", type=click.Choice(["en", "es"]), default=None)
@click.option("--stt-mode", type=click.Choice(["fast", "quality"]), default=None)
@click.option("--whisper-model", type=str, default=None)
@click.option("--log-level", type=str, default="info")
def main(host, port, language, stt_mode, whisper_model, log_level):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    settings = AppSettings()

    # CLI overrides
    if language:
        settings.language = language
    if stt_mode:
        settings.stt_mode = stt_mode
    if whisper_model:
        settings.whisper_model = whisper_model

    configure(settings)

    run_host = host or settings.host
    run_port = port or settings.port

    logging.getLogger(__name__).info(
        "Starting server at http://%s:%d (language=%s, stt=%s)",
        run_host, run_port, settings.language, settings.stt_mode,
    )

    uvicorn.run(
        app,
        host=run_host,
        port=run_port,
        log_level=log_level.lower(),
    )


if __name__ == "__main__":
    main()
