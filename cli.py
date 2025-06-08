#!/usr/bin/env python3
"""
CLI application for processing prompts using RAG with DeepSeek.
"""

import sys
import argparse
import logging
from abc import ABC, abstractmethod
from typing import Optional, List

from chatbot.rag.rag_engine import RAGEngine
from chatbot.rag.config import RETRIEVAL_K, RETRIEVAL_THRESHOLD

# Configure logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('chromadb.telemetry').setLevel(logging.WARNING)


class PromptProcessor(ABC):
    """Abstract base class for processing prompts."""

    @abstractmethod
    def process(self, prompt: str) -> str:
        """Process the given prompt and return a response."""
        pass


class RAGPromptProcessor(PromptProcessor):
    """Concrete implementation of PromptProcessor using RAG engine."""

    def __init__(self, rag_engine: RAGEngine, k: int = RETRIEVAL_K,
                    score_threshold: float = RETRIEVAL_THRESHOLD):
        """
        Initialize the processor with a RAG engine.

        Args:
            rag_engine: The RAG engine to use for processing prompts.
            k: Number of documents to retrieve.
            score_threshold: Similarity threshold for retrieval.
        """
        self.rag_engine = rag_engine
        self.k = k
        self.score_threshold = score_threshold

    def process(self, prompt: str) -> str:
        """
        Process the prompt using the RAG engine.

        Args:
            prompt: User prompt to process.

        Returns:
            Response from the RAG engine.
        """
        try:
            return self.rag_engine.ask(
                prompt,
                k=self.k,
                score_threshold=self.score_threshold
            )
        except ValueError as e:
            logger.error('Error processing prompt: %s', e)
            return f'Error: {e}'


class ArgumentParser:
    """Responsible for parsing command line arguments."""

    def __init__(self):
        """Initialize the argument parser with required configuration."""
        self.parser = argparse.ArgumentParser(
            description='Process prompts using RAG with DeepSeek.'
        )
        self._configure_parser()

    def _configure_parser(self) -> None:
        """Configure the argument parser with required arguments."""
        self.parser.add_argument(
            'prompt',
            nargs='*',
            help='The prompt to process'
        )

    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        return self.parser.parse_args()


class CLIApplication:
    """Main CLI application class that orchestrates the process."""

    def __init__(self, prompt_processor: PromptProcessor):
        """
        Initialize the CLI application.

        Args:
            prompt_processor: Component responsible for processing prompts.
        """
        self.prompt_processor = prompt_processor
        self.arg_parser = ArgumentParser()

    def run(self) -> int:
        """
        Run the application with the given arguments.

        Returns:
            Exit code: 0 for success, non-zero for errors.
        """
        args = self.arg_parser.parse_args()

        # Get the prompt from args or stdin
        prompt = self._get_prompt(args.prompt)

        if not prompt:
            print('Error: No prompt provided. Use: python cli.py \'your question here\'')
            return 1

        try:
            # Process the prompt and print the response
            print('Procesando: ', prompt)
            response = self.prompt_processor.process(prompt)
            print('\nRespuesta:')
            print(response)
            return 0
        except Exception as e:
            logger.error('Error en la aplicación: %s', e)
            return 1

    def _get_prompt(self, prompt_args: List[str]) -> Optional[str]:
        """
        Get the prompt from args or stdin if not provided.

        Args:
            prompt_args: List of arguments that form the prompt.

        Returns:
            Assembled prompt string or None if not provided.
        """
        if prompt_args:
            return ' '.join(prompt_args)

        # Check if there's input from stdin
        if not sys.stdin.isatty():
            return sys.stdin.read().strip()

        return None


def main() -> int:
    """
    Application entry point.

    Returns:
        Exit code: 0 for success, non-zero for errors.
    """
    # Initialize the RAG engine
    rag_engine = RAGEngine()

    # Create the prompt processor with the RAG engine
    prompt_processor = RAGPromptProcessor(rag_engine)

    # Create and run the CLI application
    app = CLIApplication(prompt_processor)
    return app.run()


if __name__ == '__main__':
    sys.exit(main())
