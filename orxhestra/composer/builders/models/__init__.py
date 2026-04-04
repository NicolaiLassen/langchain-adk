"""Model provider registry and factory.

Built-in providers are registered lazily on first use.  Custom providers
can be added via :func:`register` or by passing a dotted import path as
the provider name.

Supported providers::

    anthropic, anthropic_bedrock, azure_ai, azure_openai,
    bedrock, bedrock_converse, cohere, deepseek, fireworks,
    google_anthropic_vertex, google_genai, google_vertexai,
    groq, huggingface, ibm, mistralai, nvidia, ollama,
    openai, openrouter, perplexity, together, upstage, xai
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from orxhestra.composer.errors import ComposerError

_REGISTRY: dict[str, type[BaseChatModel]] = {}

# Lazy-import specs: provider name -> (module, class_name)
# Matches LangChain's init_chat_model _BUILTIN_PROVIDERS registry.
_LAZY_PROVIDERS: dict[str, tuple[str, str]] = {
    # Anthropic
    "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    "anthropic_bedrock": ("langchain_aws", "ChatAnthropicBedrock"),
    # Azure
    "azure_ai": ("langchain_azure_ai.chat_models", "AzureAIChatCompletionsModel"),
    "azure_openai": ("langchain_openai", "AzureChatOpenAI"),
    "azure": ("langchain_openai", "AzureChatOpenAI"),
    # AWS Bedrock
    "bedrock": ("langchain_aws", "ChatBedrock"),
    "bedrock_converse": ("langchain_aws", "ChatBedrockConverse"),
    "aws_bedrock": ("langchain_aws", "ChatBedrockConverse"),
    # Cohere
    "cohere": ("langchain_cohere", "ChatCohere"),
    # DeepSeek
    "deepseek": ("langchain_deepseek", "ChatDeepSeek"),
    # Fireworks
    "fireworks": ("langchain_fireworks", "ChatFireworks"),
    # Google
    "google_anthropic_vertex": (
        "langchain_google_vertexai.model_garden",
        "ChatAnthropicVertex",
    ),
    "google_genai": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
    "google": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
    "google_vertexai": ("langchain_google_vertexai", "ChatVertexAI"),
    "vertexai": ("langchain_google_vertexai", "ChatVertexAI"),
    # Groq
    "groq": ("langchain_groq", "ChatGroq"),
    # HuggingFace
    "huggingface": ("langchain_huggingface", "ChatHuggingFace"),
    # IBM watsonx
    "ibm": ("langchain_ibm", "ChatWatsonx"),
    # Mistral
    "mistral": ("langchain_mistralai", "ChatMistralAI"),
    "mistralai": ("langchain_mistralai", "ChatMistralAI"),
    # NVIDIA NIM
    "nvidia": ("langchain_nvidia_ai_endpoints", "ChatNVIDIA"),
    # Ollama (local)
    "ollama": ("langchain_ollama", "ChatOllama"),
    # OpenAI
    "openai": ("langchain_openai", "ChatOpenAI"),
    # OpenRouter
    "openrouter": ("langchain_openrouter", "ChatOpenRouter"),
    # Perplexity
    "perplexity": ("langchain_perplexity", "ChatPerplexity"),
    # Together AI
    "together": ("langchain_together", "ChatTogether"),
    # Upstage
    "upstage": ("langchain_upstage", "ChatUpstage"),
    # xAI (Grok)
    "xai": ("langchain_xai", "ChatXAI"),
}


def register(name: str, cls: type[BaseChatModel]) -> None:
    """Register a custom model provider class.

    Parameters
    ----------
    name : str
        Provider name used in YAML ``provider:`` field.
    cls : type[BaseChatModel]
        Chat model class to register.

    Example::

        register("my_provider", MyCustomChatModel)
    """
    _REGISTRY[name] = cls


def _resolve_provider(provider: str) -> type[BaseChatModel]:
    """Look up a provider class, lazy-importing built-ins as needed.

    Parameters
    ----------
    provider : str
        Registered name or dotted import path.

    Returns
    -------
    type[BaseChatModel]
        The resolved chat model class.

    Raises
    ------
    ComposerError
        If the provider cannot be imported.
    """
    if provider in _REGISTRY:
        return _REGISTRY[provider]

    if provider in _LAZY_PROVIDERS:
        module_path, class_name = _LAZY_PROVIDERS[provider]
        import importlib

        module = importlib.import_module(module_path)
        cls: type[BaseChatModel] = getattr(module, class_name)
        _REGISTRY[provider] = cls
        return cls

    # Treat as a dotted import path to a custom class.
    from orxhestra.composer.builders.tools import import_object

    cls = import_object(provider)
    _REGISTRY[provider] = cls
    return cls




def create(provider: str, name: str, **kwargs: Any) -> BaseChatModel:
    """Create a ``BaseChatModel`` from a provider name and model name.

    Parameters
    ----------
    provider : str
        A registered name (``"openai"``, ``"anthropic"``, ``"google"``), or
        a fully-qualified class import path.
    name : str
        Model name passed to the provider constructor.
    **kwargs : Any
        Extra keyword arguments forwarded to the model constructor.

    Returns
    -------
    BaseChatModel
        Instantiated chat model.
    """
    try:
        cls = _resolve_provider(provider)
    except Exception as exc:
        msg = f"Failed to load model provider '{provider}': {exc}"
        raise ComposerError(msg) from exc
    return cls(model=name, **kwargs)
