from dataclasses import dataclass, field


@dataclass
class OpenApiTTSHandlerArguments:
    open_api_tts_endpoint: str = field(
        default="https://api.openai.com/v1",
        metadata={
            "help": "The endpoint for the OpenAI API. Default is 'https://api.openai.com/v1'."
        },
    )