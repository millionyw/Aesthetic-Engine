import argparse
import subprocess


def download(url: str, output: str):
    command = ["gallery-dl", "--directory", output, url]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if process.stdout:
        for line in process.stdout:
            print(line.rstrip())
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"gallery-dl failed with exit code {process.returncode}")


def normalize_target(value: str):
    target = value.strip()
    if not target:
        return ""
    if target.startswith("http://") or target.startswith("https://"):
        return target
    handle = target.lstrip("@")
    return f"https://twitter.com/{handle}"


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url")
    group.add_argument("--file")
    parser.add_argument("--output", default="./data/raw_images")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.url:
        download(normalize_target(args.url), args.output)
        return
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            target = normalize_target(line)
            if not target:
                continue
            download(target, args.output)


if __name__ == "__main__":
    main()
