# The fine-tuned model's system prompt. Shared by build_dataset.py and train_modal.py; changing it means retraining.
SYSTEM = ("Write exactly n different natural Hebrew sentences, unpointed, 6 to 18 words each, that each use the term "
          "once, as given, in the meaning the context describes. The sentences are in the first person, and every "
          "first-person form agrees with speaker_gender (m = the \"I\" is a man, f = a woman); other people keep "
          "their own gender. Reply with a JSON array of strings.")
