from llama_index.core.tools import FunctionTool
import os 

note_file = os.path.join("data", "notes.txt")


def save_note(note:str):
    if not os.path.exists(note_file):
        with open(note_file, "w"):
            pass

    with open(note_file, "a") as f:
        f.write(note + "\n")

    return "note saved"

note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name = "note_saver",
    description="Save a note to a file",
)