"""Prompt normalization service for improving writing descriptions."""

import re
from typing import Dict, List, Optional


class PromptNormalizationService:
    """Service for normalizing and improving text prompts for image editing."""

    def __init__(self):
        """Initialize the prompt normalization service."""
        self.prompt_examples = {
            "simple": [
                "The {objects} are {action}. They are {location}. {additional_details}.",
                "There are {count} {objects}. They {action}. {location_description}.",
            ],
            "detailed": [
                "The image shows {objects} that are {action}. {size_description} {location_description}. {emotional_state}.",
                "{objects} are {action} in {location}. {detailed_description} {context_description}.",
            ],
            "narrative": [
                "In the scene, {objects} are {action}. {story_element} {emotional_description} {setting_description}.",
                "The photograph captures {objects} {action}. {narrative_element} {mood_description}.",
            ],
            "artistic": [
                "This {art_form} depicts {objects} in a state of {action}. {artistic_description} {composition_description} {emotional_impact}.",
                "The composition features {objects} {action}, creating {artistic_effect}. {visual_elements} {overall_impression}.",
            ],
        }

    def normalize_prompt(
        self, text: str, style: str = "detailed", max_length: Optional[int] = None
    ) -> str:
        """Normalize and improve a text prompt.

        Args:
            text: Input text to normalize
            style: Style of prompt ("simple", "detailed", "narrative", "artistic")
            max_length: Maximum length of output text

        Returns:
            Normalized prompt text
        """
        # Clean the text
        cleaned_text = self._clean_text(text)

        # Apply style improvements
        improved_text = self._improve_style(cleaned_text, style)

        # Truncate if needed
        if max_length and len(improved_text) > max_length:
            improved_text = self._truncate_smartly(improved_text, max_length)

        return improved_text

    def generate_variations(self, base_prompt: str, count: int = 3) -> List[str]:
        """Generate variations of a base prompt.

        Args:
            base_prompt: Base prompt to create variations from
            count: Number of variations to generate

        Returns:
            List of prompt variations
        """
        variations = []
        styles = ["simple", "detailed", "narrative", "artistic"]

        for i in range(count):
            style = styles[i % len(styles)]
            variation = self.normalize_prompt(base_prompt, style)
            variations.append(variation)

        return variations

    def extract_key_elements(self, text: str) -> Dict[str, List[str]]:
        """Extract key elements from text for prompt building.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with extracted elements (objects, actions, locations, etc.)
        """
        elements = {
            "objects": [],
            "actions": [],
            "locations": [],
            "adjectives": [],
            "emotions": [],
        }

        # Simple keyword extraction (in a real implementation,
        # you might use NLP libraries like spaCy or NLTK)
        words = text.lower().split()

        # Define some basic categories
        object_keywords = [
            "cat",
            "dog",
            "car",
            "house",
            "tree",
            "person",
            "book",
            "table",
            "chair",
        ]
        action_keywords = [
            "sleeping",
            "running",
            "sitting",
            "lying",
            "standing",
            "walking",
            "eating",
        ]
        location_keywords = [
            "on",
            "in",
            "at",
            "under",
            "above",
            "beside",
            "sofa",
            "bed",
            "table",
            "floor",
        ]
        emotion_keywords = [
            "happy",
            "sad",
            "relaxed",
            "excited",
            "peaceful",
            "content",
            "comfortable",
        ]

        for word in words:
            if any(obj in word for obj in object_keywords):
                elements["objects"].append(word)
            elif any(action in word for action in action_keywords):
                elements["actions"].append(word)
            elif any(loc in word for loc in location_keywords):
                elements["locations"].append(word)
            elif any(emotion in word for emotion in emotion_keywords):
                elements["emotions"].append(word)

        return elements

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Ensure proper sentence endings
        if not text.endswith("."):
            text += "."

        # Capitalize first letter
        text = text[0].upper() + text[1:] if text else text

        return text

    def _improve_style(self, text: str, style: str) -> str:
        """Improve text based on specified style."""
        if style == "simple":
            return self._simplify_text(text)
        elif style == "detailed":
            return self._add_details(text)
        elif style == "narrative":
            return self._make_narrative(text)
        elif style == "artistic":
            return self._make_artistic(text)
        else:
            return text

    def _simplify_text(self, text: str) -> str:
        """Simplify text to basic descriptions."""
        # Replace complex words with simpler ones
        replacements = {
            "relaxing": "resting",
            "comfortable": "cozy",
            "positioned": "placed",
            "demonstrates": "shows",
        }

        for complex_word, simple_word in replacements.items():
            text = text.replace(complex_word, simple_word)

        return text

    def _add_details(self, text: str) -> str:
        """Add descriptive details to text."""
        # Add descriptive adjectives where appropriate
        enhancements = {
            "cat": "furry cat",
            "sofa": "comfortable sofa",
            "sleeping": "peacefully sleeping",
            "lying": "casually lying",
        }

        for word, enhanced in enhancements.items():
            if word in text.lower() and enhanced not in text.lower():
                text = text.replace(word, enhanced)

        return text

    def _make_narrative(self, text: str) -> str:
        """Convert text to narrative style."""
        narrative_starters = [
            "In this peaceful scene,",
            "The image captures a moment where",
            "This photograph shows",
            "We can see",
        ]

        if not any(starter.lower() in text.lower() for starter in narrative_starters):
            starter = narrative_starters[0]
            text = f"{starter} {text.lower()}"

        return text

    def _make_artistic(self, text: str) -> str:
        """Convert text to artistic style."""
        artistic_phrases = {
            "shows": "depicts",
            "picture": "composition",
            "image": "visual narrative",
            "comfortable": "harmonious",
            "peaceful": "serene",
        }

        for basic, artistic in artistic_phrases.items():
            text = text.replace(basic, artistic)

        return text

    def _truncate_smartly(self, text: str, max_length: int) -> str:
        """Truncate text at sentence boundaries when possible."""
        if len(text) <= max_length:
            return text

        # Try to truncate at sentence boundary
        sentences = text.split(". ")
        result = ""

        for sentence in sentences:
            if len(result + sentence + ". ") <= max_length:
                result += sentence + ". "
            else:
                break

        if result:
            return result.strip()
        else:
            # Fallback to word boundary
            words = text.split()
            result = ""
            for word in words:
                if len(result + word + " ") <= max_length:
                    result += word + " "
                else:
                    break
            return result.strip() + "..."
