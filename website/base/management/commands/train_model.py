import os
import sys

from django.core.management.base import BaseCommand

WEBSITE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if WEBSITE_ROOT not in sys.path:
    sys.path.insert(0, WEBSITE_ROOT)

from model_config import MODEL_VARIANT, NUM_EPOCHS
from train_model import train_and_save


class Command(BaseCommand):
    help = (
        "Train the grape leaf model locally and save weights to website/base/. "
        "Django runserver only loads an existing .pth file — use this command to train."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--variant",
            default=MODEL_VARIANT,
            help=f"Model variant to train (default: {MODEL_VARIANT} from model_config.py)",
        )
        parser.add_argument(
            "--device",
            default="auto",
            choices=["auto", "cuda", "cpu"],
            help="Device to train on (default: auto)",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=NUM_EPOCHS,
            help=f"Maximum training epochs (default: {NUM_EPOCHS} from model_config.py)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=64,
            help="Training batch size (default: 64)",
        )
        parser.add_argument(
            "--output",
            default=None,
            help="Output .pth path relative to website/ or absolute (default: base/<variant model file>)",
        )

    def handle(self, *args, **options):
        model_path = train_and_save(
            variant=options["variant"],
            device=options["device"],
            epochs=options["epochs"],
            batch_size=options["batch_size"],
            output=options["output"],
        )
        self.stdout.write(self.style.SUCCESS(f"Saved model weights to {model_path}"))
