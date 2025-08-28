import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCRIPT = os.path.join(ROOT, "scripts", "prep_prompts_rq2_cognitive.py")


def main():
    # Delegate to the script to keep behavior identical. This file lives under experiments/cognitive.
    if not os.path.exists(SCRIPT):
        print("Missing cognitive prep script:", SCRIPT)
        sys.exit(1)
    argv = [sys.executable, SCRIPT] + sys.argv[1:]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()


