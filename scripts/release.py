#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path


def update_file(path, pattern, replacement):
    path = Path(path)
    if not path.exists():
        print(f"Error: File {path} not found.")
        sys.exit(1)

    content = path.read_text()
    new_content = re.sub(pattern, replacement, content, count=1)

    if content == new_content:
        print(f"Warning: No changes made to {path}. Pattern might not match.")
    else:
        path.write_text(new_content)
        print(f"Updated {path}")


def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Bump version and create release tag")
    parser.add_argument("version", help="New version number (e.g., 0.1.4)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not run git commmands or write files"
    )

    args = parser.parse_args()
    new_version = args.version

    # Validate version format (basic check)
    if not re.match(r"^\d+\.\d+\.\d+$", new_version):
        print(f"Warning: Version '{new_version}' does not match standard X.Y.Z format.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            sys.exit(0)

    root_dir = Path(__file__).parent.parent.absolute()

    # Update pyproject.toml
    pyproject_path = root_dir / "pyproject.toml"
    # Look for version = "..." in the [project] section mostly
    # We use a regex that matches `version = "..."`
    if not args.dry_run:
        update_file(pyproject_path, r'version = "[^"]+"', f'version = "{new_version}"')
    else:
        print(f"[Dry Run] Would update {pyproject_path} to version {new_version}")

    # Update asf/__init__.py
    init_path = root_dir / "asf" / "__init__.py"
    if not args.dry_run:
        update_file(
            init_path, r'__version__ = "[^"]+"', f'__version__ = "{new_version}"'
        )
    else:
        print(f"[Dry Run] Would update {init_path} to version {new_version}")

    # Git operations
    if not args.dry_run:
        # Check if git is clean
        status = run_command(["git", "status", "--porcelain"], cwd=root_dir)
        if status:
            # We only want to commit the specific files we changed, but if there are other changes, maybe warn?
            # Ideally we only stage our files.
            pass

        run_command(["git", "add", "pyproject.toml", "asf/__init__.py"], cwd=root_dir)
        run_command(["git", "commit", "-m", f"Release {new_version}"], cwd=root_dir)
        run_command(["git", "tag", f"v{new_version}"], cwd=root_dir)

        print("\nSuccess! Changes committed and tagged.")
        print(f"Run 'git push && git push origin v{new_version}' to publish.")
    else:
        print(f"\n[Dry Run] Would run git add, commit, and tag v{new_version}")


if __name__ == "__main__":
    main()
